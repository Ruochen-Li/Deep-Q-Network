from typing import List

import torch
from tensorboardX import SummaryWriter

from domain.jsonlookupdomain import JSONLookupDomain
from utils.sysact import SysAct, SysActionType
from modules.module import Module
from utils.beliefstate import BeliefState
from utils.useract import UserAct

from modules.policy.rl.common import DEVICE
from modules.policy.evaluation import ObjectiveReachedEvaluator
from modules.policy.rl.experience_buffer import UniformBuffer
from utils import Goal, common

from utils.logger import DiasysLogger
logger = DiasysLogger()




class RLPolicy(Module):
    """ Base class for Reinforcement Learning based policies.

    Functionallity provided includes the setup of state- and action spaces,
    conversion of `BeliefState` objects into pytorch tensors,
    updating the last performed system actions and informed entities,
    populating the experience replay buffer,
    extraction of most probable user hypothestis and candidate action 
    expansion.
    
    Output of an agent is a candidate action like
    ``inform_food`` 
    which is then populated with the most probable slot/value pair from the 
    beliefstate and database candidates by the `expand_system_action`-function 
    to become
    ``inform(slot=food,value=italian)``.

    In order to create your own policy, you can inherit from this class.
    Make sure to call the `turn_end`-function after each system turn and the
    `end_dialog`-function after each completed dialog and to overwrite the 
    `forward`-method.

    """

    def __init__(self, domain : JSONLookupDomain, subgraph = None, buffer_cls=UniformBuffer,
                 buffer_size=6000, batch_size=64, discount_gamma=0.99,
                 include_confreq=False):
        """ 
        Creates state- and action spaces, initializes experience replay 
        buffers.
        
        Arguments:
            domain {domain.jsonlookupdomain.JSONLookupDomain} -- Domain
        
        Keyword Arguments:
            subgraph {[type]} -- [see modules.Module] (default: {None})
            buffer_cls {modules.policy.rl.experience_buffer.Buffer} 
            -- [Experience replay buffer *class*, **not** an instance - will be
                initialized by this constructor!] (default: {UniformBuffer})
            buffer_size {int} -- [see modules.policy.rl.experience_buffer.
                                  Buffer] (default: {6000})
            batch_size {int} -- [see modules.policy.rl.experience_buffer.
                                  Buffer] (default: {64})
            discount_gamma {float} -- [Discount factor] (default: {0.99})
            include_confreq {bool} -- [Use confirm_request actions] 
                                       (default: {False})
        """

        super(RLPolicy, self).__init__(domain, subgraph=subgraph)
        # setup evaluator for training
        self.evaluator = ObjectiveReachedEvaluator(domain)

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.discount_gamma = discount_gamma

        self.writer = None

        # get state size
        self.state_dim = self.beliefstate_dict_to_vector(
                                BeliefState(domain)._init_beliefstate())\
                                                    .size(1)
        logger.info("state space dim: " + str(self.state_dim))

        # get system action list
        self.actions = ["inform",
                        "inform_byname", # TODO rename to 'bykey'
                        "inform_alternatives",
                        "reqmore"]
                        # TODO badaction
                        # TODO repeat not supported by user simulator
        for req_slot in self.domain.get_system_requestable_slots():
            self.actions.append('request#' + req_slot)
            self.actions.append('confirm#' + req_slot)
            self.actions.append('select#' + req_slot)
            if include_confreq:
                for conf_slot in self.domain.get_system_requestable_slots():
                    if not req_slot == conf_slot:
                        # skip case where confirm slot = request slot
                        self.actions.append('confreq#' + conf_slot + '#' +
                                                        req_slot)
        self.action_dim = len(self.actions)
        # don't include closingmsg in learnable actions
        self.actions.append('closingmsg')
        #self.actions.append("closingmsg")
        logger.info("action space dim: " + str(self.action_dim))

        self.primary_key = self.domain.get_primary_key()

        # init replay memory
        self.buffer = buffer_cls(buffer_size, batch_size, self.state_dim,
                                 discount_gamma=discount_gamma)

        self.last_sys_act = None

    def action_name(self, action_idx):
        """ Returns the action name for the specified action index """
        return self.actions[action_idx]

    def action_idx(self, action_name):
        """ Returns the action index for the specified action name """
        return self.actions.index(action_name)


    def _recurseively_extend_beliefvector(self, beliefstate, vector):
        """ Recurse through beliefstate dict and append probabilities to
            vector """

        if isinstance(beliefstate, dict):
            for key in beliefstate:
                self._recurseively_extend_beliefvector(beliefstate[key],
                                                       vector)
        if isinstance(beliefstate, float):
            vector.append(beliefstate)


    def beliefstate_dict_to_vector(self, beliefstate):
        """ Converts the beliefstate dict to a torch tensor

        Args:
            beliefstate: dict of belief (with at least beliefs and system keys)

        Returns:
            belief tensor with dimension 1 x state_dim
        """

        belief_vec = []
        # append belief state
        self._recurseively_extend_beliefvector(beliefstate['beliefs'],
                                               belief_vec)
        # append system features
        belief_vec.append(
                         float(beliefstate['system']['lastActionInformNone']))
        belief_vec.append(float(beliefstate['system']['offerHappened']))
        for dbmatch_indicator in beliefstate['system']['db_matches']:
            belief_vec.append(float(dbmatch_indicator))

        # convert to torch tensor
        return torch.tensor([belief_vec], dtype=torch.float, device=DEVICE)


    def _get_most_probable_sysreq_beliefs(self, beliefstate, consider_NONE=True,
                                   threshold = 0.7, max_results=1):
        """ Extract the most probable value for each system requestable slot

        If the most probable value for a slot does not exceed the threshold,
        then the slot will not be added to the result at all.

        Args:
            beliefstate: beliefstate dict
            consider_NONE: If True, slots where **NONE** values have the
                           highest probability will not be added to the result.
                           If False, slots where **NONE** values have the
                           highest probability will look for the best value !=
                           **NONE**.
            threshold: minimum probability to be accepted to the
            max_results: return at most #max_results best values per slot

        Returns:
            A dict with mapping from slots to a list (if max_results > 1) or
            a float (if max_results == 1) of values containing the slots which
            have at least one value whose probability exceeds the specified
            threshold.
        """

        candidates = {}
        for req_slot in self.domain.get_system_requestable_slots():
            # extract most probable value
            sorted_slot_cands = sorted(
                                    beliefstate['beliefs'][req_slot].items(),
                                    key=lambda item: item[1], reverse=True)
            if not consider_NONE:
                # filter out **NONE** values
                sorted_slot_cands = [cand for cand in sorted_slot_cands
                                          if cand[0] != '**NONE**']
            # restrict result count to specified maximum
            filtered_slot_cands = sorted_slot_cands[:max_results]
            # threshold by probabilities
            filtered_slot_cands = [slot_cand[0] for slot_cand
                                                in filtered_slot_cands
                                                if slot_cand[1] >= threshold]
            if '**NONE**' in filtered_slot_cands:
                # remove **NONE** from results
                filtered_slot_cands.remove('**NONE**')
            if len(filtered_slot_cands) > 0:
                # append results if any remain after filtering
                if max_results == 1:
                    # only float
                    candidates[req_slot] = filtered_slot_cands[0]
                else:
                    # list
                    candidates[req_slot] = filtered_slot_cands
        return candidates


    def _get_most_probable_inf_beliefs(self, beliefstate, consider_NONE=True,
                                   threshold = 0.7, max_results=1):
        """ Extract the most probable value for each system requestable slot

        If the most probable value for a slot does not exceed the threshold,
        then the slot will not be added to the result at all.

        Args:
            beliefstate: beliefstate dict
            consider_NONE: If True, slots where **NONE** values have the
                           highest probability will not be added to the result.
                           If False, slots where **NONE** values have the
                           highest probability will look for the best value !=
                           **NONE**.
            threshold: minimum probability to be accepted to the
            max_results: return at most #max_results best values per slot

        Returns:
            A dict with mapping from slots to a list (if max_results > 1) or
            a float (if max_results == 1) of values containing the slots which
            have at least one value whose probability exceeds the specified
            threshold.
        """

        candidates = {}
        for inf_slot in self.domain.get_informable_slots():
            # extract most probable value
            sorted_slot_cands = sorted(
                                    beliefstate['beliefs'][inf_slot].items(),
                                    key=lambda item: item[1], reverse=True)
            if not consider_NONE:
                # filter out **NONE** values
                sorted_slot_cands = [cand for cand in sorted_slot_cands
                                          if cand[0] != '**NONE**']
            # restrict result count to specified maximum
            filtered_slot_cands = sorted_slot_cands[:max_results]
            # threshold by probabilities
            filtered_slot_cands = [slot_cand[0] for slot_cand
                                                in filtered_slot_cands
                                                if slot_cand[1] >= threshold]
            if '**NONE**' in filtered_slot_cands:
                # remove **NONE** from results
                filtered_slot_cands.remove('**NONE**')
            if len(filtered_slot_cands) > 0:
                # append results if any remain after filtering
                if max_results == 1:
                    # only float
                    candidates[inf_slot] = filtered_slot_cands[0]
                else:
                    # list
                    candidates[inf_slot] = filtered_slot_cands
        return candidates


    def _get_requested_slots(self, beliefstate, threshold = 0.7):
        """ Returns the slots requested by the user with
            probability > threshold and slotname != **NONE** """

        candidates = []
        for req_slot, req_prob in beliefstate['beliefs']['requested'].items():
            if req_slot != '**NONE**' and req_prob > threshold:
                candidates.append(req_slot)
        return candidates


    def _remove_dontcare_slots(self, slot_value_dict):
        """ Returns a new dictionary without the slots set to dontcare """

        return {slot: value for slot, value in slot_value_dict.items()
                            if value != 'dontcare'}


    def _get_slotnames_from_actionname(self, action_name):
        """ Return the slot names of an action of format 'action_slot1_slot2_...' """
        return action_name.split('#')[1:]


    def _db_results_to_sysact(self, sys_act, constraints, db_entity):
        """ Adds values of db_entity to constraints of sys_act
            (primary key is always added).
            Omits values which are not available in database. """

        for constraint_slot in constraints:
            if constraints[constraint_slot] == 'dontcare' and \
               constraint_slot in db_entity and \
               db_entity[constraint_slot] != 'not available':
                # fill user dontcare with database value
                sys_act.add_value(constraint_slot, db_entity[constraint_slot])
            else:
                if constraint_slot in db_entity:
                    # fill with database value
                    sys_act.add_value(constraint_slot,
                                      db_entity[constraint_slot])
                else:
                    # slot not in db entity -> create warning
                    logger.warning("Slot " + constraint_slot +
                                   " not found in db entity " +
                                   db_entity[self.primary_key])
        # ensure primary key is included
        if not self.primary_key in sys_act.slot_values:
            sys_act.add_value(self.primary_key, db_entity[self.primary_key])


    def _expand_byconstraints(self, beliefstate):
        """ Create inform act with an entity from the database, if any matches
            could be found for the constraints, otherwise will return an inform
            act with primary key=none """
        act = SysAct()
        act.type = SysActionType.Inform

        # get constraints and query db by them
        constraints = self._get_most_probable_inf_beliefs(beliefstate,
                                                      consider_NONE=True,
                                                      threshold = 0.7,
                                                      max_results=1)
        db_matches = self.domain.find_entities(constraints)
        if len(db_matches) == 0:
            # no matching entity found -> return inform with primary key=none
            # and other constraints
            filtered_slot_values = self._remove_dontcare_slots(constraints)
            filtered_slots = common.numpy.random.choice(
                            list(filtered_slot_values.keys()), 
                            min(5, len(filtered_slot_values)), replace=False)
            for slot in filtered_slots:
                if not slot == 'name':
                    act.add_value(slot, filtered_slot_values[slot])
            act.add_value(self.primary_key, 'none')
        else:
            # match found -> return its name
            # if > 1 match and matches contain last informed entity,
            # stick to this
            match = [db_match for db_match in db_matches
                              if db_match[self.primary_key] == \
                               beliefstate['system']['lastInformedPrimKeyVal']]
            if len(match) == 0:
                # none matches last informed venue -> pick first result
                # match = db_matches[0]
                match = common.random.choice(db_matches)
            else:
                assert len(match) == 1
                match = match[0]
            # fill act with values from db
            self._db_results_to_sysact(act, constraints, match)

        return act


    def _expand_request(self, action_name):
        """ Expand request_*slot* action """
        act = SysAct()
        act.type = SysActionType.Request
        req_slot = self._get_slotnames_from_actionname(action_name)[0]
        act.add_value(req_slot)
        return act


    def _expand_select(self, action_name, beliefstate):
        """ Expand select_*slot* action """
        act = SysAct()
        act.type = SysActionType.Select
        sel_slot = self._get_slotnames_from_actionname(action_name)[0]
        most_likely_choice = self._get_most_probable_sysreq_beliefs(beliefstate,
                                    consider_NONE=False, threshold=0.0,
                                    max_results=2)
        first_value = most_likely_choice[sel_slot][0]
        second_value = most_likely_choice[sel_slot][1]
        act.add_value(sel_slot, first_value)
        act.add_value(sel_slot, second_value)
        return act


    def _expand_confirm(self, action_name, beliefstate):
        """ Expand confirm_*slot* action """
        act = SysAct()
        act.type = SysActionType.Confirm
        conf_slot = self._get_slotnames_from_actionname(action_name)[0]
        candidates = self._get_most_probable_inf_beliefs(beliefstate,
                                    consider_NONE=False, threshold=0.0,
                                    max_results=1)
        conf_value = candidates[conf_slot]
        act.add_value(conf_slot, conf_value)
        return act


    def _expand_confreq(self, action_name, beliefstate):
        """ Expand confreq_*confirmslot*_*requestslot* action """
        act = SysAct()
        act.type = SysActionType.ConfirmRequest
        # first slot name is confirmation, second is request
        slots = self._get_slotnames_from_actionname(action_name)
        conf_slot = slots[0]
        req_slot = slots[1]

        # get value that needs confirmation
        candidates = self._get_most_probable_inf_beliefs(beliefstate,
                                    consider_NONE=False, threshold=0.0,
                                    max_results=1)
        conf_value = candidates[conf_slot]
        act.add_value(conf_slot, conf_value)
        # add request slot
        act.add_value(req_slot)
        return act

    def _expand_informbyname(self, beliefstate):
        """ Expand inform_byname action """
        act = SysAct()
        act.type = SysActionType.InformByName

        # get most probable entity primary key
        primkeyval = sorted(beliefstate['beliefs'][self.primary_key].items(),
                      key=lambda item: item[1], reverse=True)[0][0]
        if primkeyval == '**NONE**':
            # try to use previously informed name instead
            primkeyval = beliefstate['system']['lastInformedPrimKeyVal']
        # find db entry by primary key
        constraints = self._get_most_probable_inf_beliefs(beliefstate,
                                                    consider_NONE=True,
                                                    threshold = 0.7,
                                                    max_results=1)
        db_matches = self.domain.find_entities({self.primary_key: primkeyval})
        # db_matches = om.query_db_byslotvalues(self.domain,
        #                                       {**constraints, self.primary_key: primkeyval}) # NOTE usually not needed to give all constraints (shouldn't make a difference)
        if not db_matches:
            # act.add_value(self.primary_key, 'none') # TODO testing code
            # return act
            # select random entity if none could be found
            primkeyvals = self.domain.get_possible_values(self.primary_key)
            primkeyval = common.random.choice(primkeyvals)
            db_matches = self.domain.find_entities(constraints) # use knowledge from current belief state
            # db_matches = om.query_db_byslotvalues(self.domain, 
            #                                     {self.primary_key: primkeyval})
        if not db_matches:
            # no results found
            filtered_slot_values = self._remove_dontcare_slots(constraints)
            for slot in common.numpy.random.choice(list(filtered_slot_values.keys()), min(5, len(filtered_slot_values)), replace=False):
                act.add_value(slot, filtered_slot_values[slot])
            act.add_value(self.primary_key, 'none')
            return act

        # select random match
        db_match = common.random.choice(db_matches)

        # get slots requested by user
        usr_requests = self._get_requested_slots(beliefstate, threshold = 0.0)
        # remove primary key (to exlude from minimum number) since it is added anyway at the end
        if self.primary_key in usr_requests:
            usr_requests.remove(self.primary_key)
        if usr_requests:
            # add user requested values into system act using db result
            for req_slot in common.numpy.random.choice(usr_requests, min(4, len(usr_requests)), replace=False):
                if req_slot in db_match:
                    act.add_value(req_slot, db_match[req_slot])
                else:
                    act.add_value(req_slot, 'none')
        else:
            constraints = self._remove_dontcare_slots(constraints)
            if constraints:
                for inform_slot in common.numpy.random.choice(list(constraints.keys()), min(4, len(constraints)), replace=False):
                    value = db_match[inform_slot]
                    act.add_value(inform_slot, value)
            else:
                # add random slot and value if no user request was detected
                usr_requestable_slots = set(self.domain.get_requestable_slots())
                usr_requestable_slots.remove(self.primary_key)
                random_slot = common.random.choice(list(usr_requestable_slots))
                value = db_match[random_slot]
                act.add_value(random_slot, value)
        # ensure entity primary key is included
        if not self.primary_key in act.slot_values:
            act.add_value(self.primary_key, db_match[self.primary_key])

        return act


    def _expand_informbyalternatives(self, beliefstate):
        """ Expand inform_byalternatives action """
        act = SysAct()
        act.type = SysActionType.InformByAlternatives
        # get set of all previously informed primary key values
        informedPrimKeyValsSinceNone = set(
                              beliefstate['system']['informedPrimKeyValsSinceNone'])
        candidates = self._get_most_probable_inf_beliefs(beliefstate,
                                                     consider_NONE=True,
                                                     threshold=0.7,
                                                     max_results=1)
        filtered_slot_values = self._remove_dontcare_slots(candidates)
        # query db by constraints
        db_matches = self.domain.find_entities(candidates)
        if not db_matches:
            # no results found
            for slot in common.numpy.random.choice(list(filtered_slot_values.keys()), min(5, len(filtered_slot_values)), replace=False):
                act.add_value(slot, filtered_slot_values[slot])
            act.add_value(self.primary_key, 'none')
            return act

        # don't inform about already informed entities
        # -> exclude primary key values from informedPrimKeyValsSinceNone
        for db_match in db_matches:
            if db_match[self.primary_key] not in informedPrimKeyValsSinceNone:
                # new entity!
                # TODO move to _db_results_to_sysact method (including maximum inform slot-values)
                # get slots requested by user
                usr_requests = self._get_requested_slots(beliefstate, threshold = 0.0)
                if self.primary_key in usr_requests:
                    usr_requests.remove(self.primary_key)
                if usr_requests:
                    # add user requested values into system act using db result
                    for req_slot in common.numpy.random.choice(usr_requests, min(4, len(usr_requests)), replace=False):
                        if req_slot in db_match:
                            act.add_value(req_slot, db_match[req_slot])
                        else:
                            act.add_value(req_slot, 'none')
                # add slots for which the value != 'dontcare'
                for slot in common.numpy.random.choice(list(filtered_slot_values.keys()), min(4-(len(act.slot_values)), len(filtered_slot_values)), replace=False):
                    act.add_value(slot, filtered_slot_values[slot])
                additional_constraints = {}
                for slot, value in candidates.items():
                    if len(act.slot_values) < 4 and value == 'dontcare':
                        additional_constraints[slot] = value
                self._db_results_to_sysact(act, additional_constraints, db_match)
                
                return act

        # no alternatives found (that were not already mentioned)
        act.add_value(self.primary_key, 'none')
        return act


    def _expand_bye(self):
        """ Expand bye action """
        act = SysAct()
        act.type = SysActionType.Bye
        return act


    def _expand_reqmore(self):
        """ Expand reqmore action """
        act = SysAct()
        act.type = SysActionType.RequestMore
        return act


    def expand_system_action(self, action_idx, beliefstate):
        """ Expands an action index to a real sytem act """

        action_name = self.action_name(action_idx)
        if action_name == 'inform':
            return self._expand_byconstraints(beliefstate)
        elif 'request#' in action_name:
            return self._expand_request(action_name)
        elif 'select#' in action_name:
            return self._expand_select(action_name, beliefstate)
        elif 'confirm#' in action_name:
            return self._expand_confirm(action_name, beliefstate)
        elif 'confreq#' in action_name:
            return self._expand_confreq(action_name, beliefstate)
        elif action_name == 'inform_byname':
            return self._expand_informbyname(beliefstate)
        elif action_name == 'inform_alternatives':
            return self._expand_informbyalternatives(beliefstate)
        elif action_name == 'closingmsg':
            return self._expand_bye()
        elif action_name == 'repeat':
            return self.last_sys_act
        elif action_name == 'reqmore':
            return self._expand_reqmore()
        else:
            logger.warning("RL POILICY: system action not supported: " +
                           action_name)
        # TODO restart: not supported by PyDial
        # -> check if user simulator supports this

        return None


    def _update_system_belief(self, beliefstate, sys_act):
        """ Update the system's belief state features """

        # check if system informed an entity primary key vlaue this turn
        # (and remember it), otherwise, keep previous one
        # reset, if primary key value == none (found no matching entities)
        beliefstate['system']['lastActionInformNone'] = False
        beliefstate['system']['offerHappened'] = False
        informed_names = sys_act.get_values(self.primary_key)
        if len(informed_names) > 0 and len(informed_names[0]) > 0:
            beliefstate['system']['offerHappened'] = True
            beliefstate['system']['lastInformedPrimKeyVal'] = informed_names[0]
            if informed_names[0] == 'none':
                beliefstate['system']['informedPrimKeyValsSinceNone'] = []
                beliefstate['system']['lastActionInformNone'] = True
            else:
                beliefstate['system']['informedPrimKeyValsSinceNone'].append(
                                                            informed_names[0])

        # check how many db entities match the current constraints
        candidates = self._get_most_probable_inf_beliefs(beliefstate,
                                                     consider_NONE=True,
                                                     threshold=0.7,
                                                     max_results=1)
        constraints = self._remove_dontcare_slots(candidates)
        db_matches = self.domain.find_entities(constraints)
        candidate_count = len(db_matches)
        beliefstate['system']['db_matches'][0] = (candidate_count == 0)
        beliefstate['system']['db_matches'][1] = (candidate_count == 1)
        beliefstate['system']['db_matches'][2] = (2 <= candidate_count <= 4)
        beliefstate['system']['db_matches'][3] = (candidate_count > 4)

        # check if matching db entities could be discriminated by more
        # information from user
        discriminable = False
        if len(db_matches) < 2:
            discriminable = True
        else:
            dontcare_slots = set(candidates.keys()) - set(constraints.keys())
            informable_slots = self.domain.get_informable_slots() - \
                                                          set(self.primary_key)
            for informable_slot in informable_slots:
                if informable_slot not in dontcare_slots:
                    # this slot could be used to gather more information
                    db_values_for_slot = set()
                    for db_match in db_matches:
                        db_values_for_slot.add(db_match[informable_slot])
                    if len(db_values_for_slot) > 1:
                        # at least 2 different values for slot
                        # ->can use this slot to differentiate between entities
                        discriminable = True
                        break
        beliefstate['system']['db_matches'][4] = discriminable

        # set last system request slot s.t. BST can infer "this" reference
        # from user simulator (where user inform omits primary key slot)
        if sys_act.type == SysActionType.Request or sys_act.type == SysActionType.Select:
            beliefstate['system']['lastRequestSlot'] = list(sys_act.slot_values.keys())[0]
        elif sys_act.type == SysActionType.ConfirmRequest:
            req_slot = None
            for slot in sys_act.slot_values:
                if len(sys_act.get_values(slot)) == 0:
                    req_slot = slot
                    break
            beliefstate['system']['lastRequestSlot'] = req_slot
        else:
            beliefstate['system']['lastRequestSlot'] = None


    def turn_end(self, beliefstate, state_vector, sys_act_idx):
        """ Call this function after a turn is done by the system """

        self.last_sys_act = self.expand_system_action(sys_act_idx, beliefstate)
        # TODO COMPATIBILITY TO PYDIAL USR SIMULATOR AND NLG CURRENTLY - REMOVE LATER
        # if self.last_sys_act.type == SysActionType.InformByName:
        #     self.last_sys_act.type = SysActionType.Inform
        logger.dialog_turn("system action > " + str(self.last_sys_act))
        self._update_system_belief(beliefstate, self.last_sys_act)

        turn_reward = self.evaluator.get_turn_reward()

        if self.is_training:
            self.buffer.store(state_vector, sys_act_idx, turn_reward,
                            terminal=False)


    def _expand_hello(self):
        """ Call this function when a dialog begins """

        hello_action = SysAct()
        hello_action.type = SysActionType.Welcome
        self.last_sys_act = hello_action
        logger.dialog_turn("system action > " + str(hello_action))
        return {'sys_act': hello_action}


    def end_dialog(self, sim_goal):
        """ Call this function when a dialog ended """
        if sim_goal is None:
            # real user interaction, no simulator - don't have to evaluate
            # anything, just reset counters
            return

        final_reward, success = self.evaluator.get_final_reward(sim_goal, 
                                                                 logging=False)

        if self.is_training:
            self.buffer.store(None, None, final_reward, terminal=True)

        # if self.writer is not None:
        #     self.writer.add_scalar('buffer/items', len(self.buffer),
        #                     self.train + self.total_train_dialogs)




    def forward(self, dialog_graph, user_acts: List[UserAct] = None,
                beliefstate: BeliefState = None, sim_goal : Goal = None,
                writer : SummaryWriter = None, **kwargs) -> dict(sys_act=SysAct):
        """ Child classes have to overwrite this method """
        self.writer = writer
        return {}
