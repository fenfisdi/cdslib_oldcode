import copy
import numpy as np

from cdslib.hospitals import Hospital

dim = 2

def determine_age_group(
    population_age_groups_info: dict,
    age: float
    ):
    """
    """
    for age_group in population_age_groups_info.keys():

        age_group_dict = population_age_groups_info[age_group]

        if age_group_dict['min_age'] and age_group_dict['max_age']:
            if (age_group_dict['min_age'] <= age
            and np.floor(age) <= age_group_dict['max_age']):
                return age_group

        if not age_group_dict['min_age']:
            if np.floor(age) <= age_group_dict['max_age']:
                return age_group

        if not age_group_dict['max_age']:
            if age_group_dict['min_age'] <= age:
                return age_group


class AgentsInfo:
    """
    """
    def __init__(
        self,
        disease_states: list,
        susceptibility_groups: list,
        vulnerability_groups: list,
        dynamics_of_the_disease_states_transitions_info: dict,
        contagion_dynamics_info: dict,
        population_age_groups_info: dict,
        diagnosis_of_disease_states_by_vulnerability_group: dict,
        hospitalization_of_disease_states_by_vulnerability_group: dict,
        social_distancing_info: dict
        ):
        """
        """
        self.disease_states = disease_states
        self.susceptibility_groups = susceptibility_groups
        self.vulnerability_groups = vulnerability_groups

        # dynamics_of_the_disease_states_transitions_info
        self.disease_states_time_functions = \
            dynamics_of_the_disease_states_transitions_info[
                'disease_states_time_functions'
                ]

        self.criticality_level_of_evolution_of_disease_states = \
            dynamics_of_the_disease_states_transitions_info[
                'criticality_level_of_evolution_of_disease_states'
                ]

        self.disease_states_transitions_by_vulnerability_group = \
            dynamics_of_the_disease_states_transitions_info[
                'disease_states_transitions_by_vulnerability_group'
                ]

        self.disease_states_transitions_by_hospitalization_unavailability_by_vulnerability_group = \
            dynamics_of_the_disease_states_transitions_info[
                'disease_states_transitions_by_hospitalization_unavailability_by_vulnerability_group'
                ]

        self.disease_states_transitions_by_UCI_unavailability_by_vulnerability_group = \
            dynamics_of_the_disease_states_transitions_info[
                'disease_states_transitions_by_UCI_unavailability_by_vulnerability_group'
                ]

        # contagion_dynamics_info
        self.disease_states_transitions_by_contagion = \
            contagion_dynamics_info[
                'disease_states_transitions_by_contagion'
                ]

        self.criticality_level_of_disease_states_to_susceptibility_to_contagion = \
            contagion_dynamics_info[
                'criticality_level_of_disease_states_to_susceptibility_to_contagion'
            ]

        self.contagion_probabilities_by_susceptibility_groups = \
            contagion_dynamics_info[
                'contagion_probabilities_by_susceptibility_groups'
                ]

        self.dynamics_of_disease_states_contagion = \
            contagion_dynamics_info[
                'dynamics_of_disease_states_contagion'
                ]

        self.inmunization_level_gained_by_disease_by_vulnerability_group = \
            contagion_dynamics_info[
                'inmunization_level_gained_by_disease_by_vulnerability_group'
                ]

        # population_age_groups_info
        self.population_age_groups_info = population_age_groups_info

        self.diagnosis_of_disease_states_by_vulnerability_group = \
            diagnosis_of_disease_states_by_vulnerability_group

        # hospitalization_of_disease_states_by_vulnerability_group
        self.hospitalization_of_disease_states_by_vulnerability_group = \
            hospitalization_of_disease_states_by_vulnerability_group

        # social_distancing_info
        self.dynamics_of_alertness_of_disease_states_by_vulnerability_group = \
            social_distancing_info[
                'dynamics_of_alertness_of_disease_states_by_vulnerability_group'
                ]

        self.dynamics_of_avoidance_of_disease_states_by_vulnerability_group = \
            social_distancing_info[
                'dynamics_of_avoidance_of_disease_states_by_vulnerability_group'
                ]

class Agent:
    """
    """
    def __init__(
        self,
        agents_info: AgentsInfo,
        x: float,
        y: float,
        vmax: float,
        agent: int,
        disease_state: str,
        susceptibility_group: str,
        vulnerability_group: str,
        age: float,
        age_group: str,
        inmunization_level: float,
        quarantine_group: str,
        obedience_to_quarantine: float,
        diagnosed: bool=False
        ):
        """
        """
        # Define private atributes from AgentsInfo atributes
        for key, value in zip(
            agents_info.__dict__.keys(),
            agents_info.__dict__.values()
            ):
            setattr(self, '_Agent__' + key, value)


        # Retrieve maximum radius for trace_neighbors function
        spread_radius_list = []

        for dis_state in self.__disease_states:

            spread_radius_list.append(
                self.__dynamics_of_disease_states_contagion[
                dis_state]['spread_radius']
                )


        avoidness_radius_list = []

        for vul_group in self.__vulnerability_groups:

            for dis_state in self.__disease_states:

                avoidness_radius_list.append(
                    self.__dynamics_of_avoidance_of_disease_states_by_vulnerability_group[
                    vul_group][dis_state]['avoidness_radius']
                    )

        spread_radius_arr = np.array(spread_radius_list, dtype=np.float)
        spread_radius_arr = np.nan_to_num(spread_radius_arr, nan=-np.inf)
        max_spread_radius = spread_radius_arr.max()

        avoidness_radius_arr = np.array(avoidness_radius_list, dtype=np.float)
        avoidness_radius_arr = np.nan_to_num(avoidness_radius_arr, nan=-np.inf)
        max_avoidness_radius = avoidness_radius_arr.max()

        self.tracing_radius = np.maximum(
            max_spread_radius,
            max_avoidness_radius
            )

        # Define vmax
        self.vmax = vmax

        # Define agent label
        self.agent: int = agent

        self.age: float = age

        self.age_group: str = age_group

        # Define position
        self.x: float = x
        self.y: float = y

        # Define initial velocity
        self.initialize_velocity()

        # Initialize disease_state, susceptibility_group,
        # vulnerability_group and diagnosis state 
        self.disease_state: str = disease_state

        self.susceptibility_group: str = susceptibility_group

        self.vulnerability_group: str = vulnerability_group

        self.waiting_diagnosis: bool = False

        self.time_waiting_diagnosis: int = None

        self.diagnosed: bool = diagnosed

        self.time_after_being_diagnosed: int = None

        self.needs_hospitalization: bool = False

        self.is_hospitalized: bool = False

        self.hospital_label: str = None

        self.time_waiting_hospitalization: int = None

        self.needs_UCI: bool = False

        self.is_in_UCI: bool = False

        self.time_waiting_UCI: int = None

        self.infected_by: list = []
        self.infected_in_step: int = None
        self.infected_info: dict = {}

        self.inmunization_level: float = inmunization_level

        self.quarantine_group: str = quarantine_group
        self.obedience_to_quarantine: float = obedience_to_quarantine
        self.quarantine_state: str = 'Not in quarantine'

        self.susceptible_neighbors: list = []

        self.infected_spreader_neighbors: list = []

        self.infected_non_spreader_neighbors: list = []

        self.inmune_neighbors: list = []

        self.total_neighbors: list = []

        self.avoidable_neighbors: list = []

        self.alertness: bool = False

        self.alerted_by: list = []

        self.disease_state_time: int = 0

        self.live_state: str = 'alive'

        # Determine state time
        self.determine_disease_state_time()

        # Determine times infected
        self.is_infected: bool
        self.times_infected: int

        if self.__dynamics_of_disease_states_contagion[self.disease_state]['is_infected']:
            self.is_infected = True
            self.times_infected = 1
        else:
            self.is_infected = False
            self.times_infected = 0


    def getstate(self):
        """
        """
        agent_dict = copy.deepcopy(self.__dict__)

        # Remove private attributes
        remove_list = [key for key in agent_dict.keys() if '_Agent__' in key]

        for key in remove_list:
            del agent_dict[key]

        return agent_dict


    def getkeys(self):
        """
        """
        agent_dict = self.getstate()

        return agent_dict.keys()


    def initialize_velocity(
        self
        ):
        """
        """
        # Define velocity between (0 , vmax)
        v = self.vmax * np.random.random_sample()
        theta = 2. * np.pi * np.random.random_sample()

        self.vx, self.vy = [v * np.cos(theta), v * np.sin(theta)]


    def determine_disease_state_time(self):
        """
        """
        if self.__disease_states_time_functions[
            self.vulnerability_group][self.disease_state]['time_function']:

            self.disease_state_max_time = \
                self.__disease_states_time_functions[
                self.vulnerability_group][self.disease_state]['time_function']()

        else:
            self.disease_state_max_time = None


    def disease_state_transition(
        self,
        dt: int,
        hospitals: dict,
        hospitals_spatial_tree,
        hospitals_labels
        ):
        """
        """
        self.disease_state_time += dt
        previous_disease_state = self.disease_state

        if (self.disease_state_max_time
        and self.disease_state_time >= self.disease_state_max_time):

            # Verify: becomes into ? ... Throw the dice
            dice = np.random.random_sample()

            cummulative_probability = 0. + self.inmunization_level

            for (probability, becomes_into_disease_state) in sorted(
                zip(
                    self.__disease_states_transitions_by_vulnerability_group[
                    self.vulnerability_group][self.disease_state]['transition_probability'],
                    self.__disease_states_transitions_by_vulnerability_group[
                    self.vulnerability_group][self.disease_state]['becomes_into']
                    ),
                # In order to use criticality_level
                # pair = (probability, becomes_into_disease_state)
                key=lambda pair: self.__criticality_level_of_evolution_of_disease_states[pair[1]]
                ):

                cummulative_probability += probability

                if dice <= cummulative_probability:

                    self.update_infection_status(becomes_into_disease_state)

                    self.disease_state = becomes_into_disease_state

                    if becomes_into_disease_state == 'dead':
                        # Agent died by disease
                        self.live_state = 'dead by disease'

                    self.update_diagnosis_state(dt, previous_disease_state)

                    hospitals = self.update_hospitalization_state(
                        dt,
                        hospitals,
                        hospitals_spatial_tree,
                        hospitals_labels
                        )

                    self.disease_state_time = 0

                    self.determine_disease_state_time()

                    break

        else:
            self.update_diagnosis_state(dt, previous_disease_state)

            hospitals = self.update_hospitalization_state(
                dt,
                hospitals,
                hospitals_spatial_tree,
                hospitals_labels
                )

        return hospitals


    def disease_state_transition_by_contagion(
        self,
        step: int,
        dt: int,
        spatial_trees_by_disease_state: dict,
        agents_indices_by_disease_state: dict
        ):
        """
        """
        # Agent can get infected ?
        if self.__dynamics_of_disease_states_contagion[
            self.disease_state]['can_get_infected']:

            # Retrieve agent location
            agent_location = [self.x, self.y]

            # List to save who infected the agent
            infected_by = []

            # Cycle through each spreader to see if the agent gets 
            # infected by the spreader
            for disease_state in self.__disease_states:

                if (self.__dynamics_of_disease_states_contagion[
                    disease_state]['can_spread']
                and spatial_trees_by_disease_state[disease_state]):

                    # Detect if any spreader is inside a distance equal to
                    # the corresponding spread_radius
                    points_inside_radius = \
                        spatial_trees_by_disease_state[disease_state].query_ball_point(
                            agent_location,
                            self.__dynamics_of_disease_states_contagion[
                                disease_state]['spread_radius']
                            )

                    spreaders_indices_inside_radius = \
                        agents_indices_by_disease_state[
                            disease_state][points_inside_radius]

                    # If self.agent in spreaders_indices_inside_radius,
                    # then remove it
                    if self.agent in spreaders_indices_inside_radius:

                        spreaders_indices_inside_radius = np.setdiff1d(
                            spreaders_indices_inside_radius,
                            self.agent
                            )

                    # Calculate joint probability for contagion
                    joint_probability = \
                        (1.0 - self.inmunization_level) \
                        * self.__contagion_probabilities_by_susceptibility_groups[
                            self.susceptibility_group] \
                        * self.__dynamics_of_disease_states_contagion[
                            disease_state]['spread_probability']

                    # Check if got infected
                    for spreader_agent_index in spreaders_indices_inside_radius:
                        # Throw the dice
                        dice = np.random.random_sample()

                        if dice <= joint_probability:
                            # Got infected !!!
                            # Save who infected the agent
                            infected_by.append(spreader_agent_index)


            if len(infected_by) is not 0:

                self.infected_by = infected_by
                self.infected_in_step = step
                self.infected_info[step] = infected_by

                # Verify: becomes into ? ... Throw the dice
                dice = np.random.random_sample()

                cummulative_probability = 0. + self.inmunization_level

                for (probability, becomes_into_disease_state) in sorted(
                    zip(
                        self.__disease_states_transitions_by_contagion[
                            self.disease_state]['transition_probability'],
                        self.__disease_states_transitions_by_contagion[
                            self.disease_state]['becomes_into']
                        ),
                    # In order to use criticality_level
                    # pair = (probability, becomes_into_disease_state)
                    key=lambda pair: self.__criticality_level_of_disease_states_to_susceptibility_to_contagion[pair[1]]
                    ):
                    cummulative_probability += probability

                    if dice <= cummulative_probability:

                        self.update_infection_status(becomes_into_disease_state)

                        previous_disease_state = self.disease_state

                        self.disease_state = becomes_into_disease_state

                        self.update_diagnosis_state(dt, previous_disease_state)

                        self.disease_state_time = 0

                        self.determine_disease_state_time()

                        break
        else:
            # Agent cannot get infected
            pass


    def disease_states_transitions_by_hospitalization_unavailability(self):
        """
        """

        if self.__disease_states_transitions_by_hospitalization_unavailability_by_vulnerability_group[
        self.vulnerability_group][self.disease_state]['max_time_waiting_hospitalization']:

            if self.time_waiting_hospitalization >= self.__disease_states_transitions_by_hospitalization_unavailability_by_vulnerability_group[
            self.vulnerability_group][self.disease_state]['max_time_waiting_hospitalization']:

                # Verify: becomes into ? ... Throw the dice
                dice = np.random.random_sample()

                cummulative_probability = 0. + self.inmunization_level

                for (probability, becomes_into_disease_state) in sorted(
                    zip(
                        self.__disease_states_transitions_by_hospitalization_unavailability_by_vulnerability_group[
                        self.vulnerability_group][self.disease_state]['transition_probability'],
                        self.__disease_states_transitions_by_hospitalization_unavailability_by_vulnerability_group[
                        self.vulnerability_group][self.disease_state]['becomes_into']
                        ),
                    # In order to use criticality_level
                    # pair = (probability, becomes_into_disease_state)
                    key=lambda pair: self.__criticality_level_of_evolution_of_disease_states[pair[1]]
                    ):

                    cummulative_probability += probability

                    if dice <= cummulative_probability:

                        self.update_infection_status(becomes_into_disease_state)

                        self.disease_state = becomes_into_disease_state

                        if becomes_into_disease_state == 'dead':
                            # Agent died by disease
                            self.live_state = 'dead by disease'

                        self.disease_state_time = 0

                        self.determine_disease_state_time()

                        break


    def disease_states_transitions_by_UCI_unavailability(self):
        """
        """

        if self.__disease_states_transitions_by_UCI_unavailability_by_vulnerability_group[
        self.vulnerability_group][self.disease_state]['max_time_waiting_UCI']:

            if self.time_waiting_UCI >= self.__disease_states_transitions_by_UCI_unavailability_by_vulnerability_group[
            self.vulnerability_group][self.disease_state]['max_time_waiting_UCI']:

                # Verify: becomes into ? ... Throw the dice
                dice = np.random.random_sample()

                cummulative_probability = 0. + self.inmunization_level

                for (probability, becomes_into_disease_state) in sorted(
                    zip(
                        self.__disease_states_transitions_by_UCI_unavailability_by_vulnerability_group[
                        self.vulnerability_group][self.disease_state]['transition_probability'],
                        self.__disease_states_transitions_by_UCI_unavailability_by_vulnerability_group[
                        self.vulnerability_group][self.disease_state]['becomes_into']
                        ),
                    # In order to use criticality_level
                    # pair = (probability, becomes_into_disease_state)
                    key=lambda pair: self.__criticality_level_of_evolution_of_disease_states[pair[1]]
                    ):

                    cummulative_probability += probability

                    if dice <= cummulative_probability:

                        self.update_infection_status(becomes_into_disease_state)

                        self.disease_state = becomes_into_disease_state

                        if becomes_into_disease_state == 'dead':
                            # Agent died by disease
                            self.live_state = 'dead by disease'

                        self.disease_state_time = 0

                        self.determine_disease_state_time()

                        break


    def determine_age_group(self):
        """
        """
        self.age_group = determine_age_group(
            self.__population_age_groups_info,
            self.age
            )


    def age_and_death_verification(
        self,
        dt_scale_in_years: float
        ):
        """
        """
        dead = False

        if self.live_state == 'dead by disease':
            dead = True
        else:
            #=============================================
            # Increment age
            self.age += dt_scale_in_years

            # Determine age group
            self.determine_age_group()

            #=============================================
            # Verify: natural death ? ... Throw the dice
            dice = np.random.random_sample()

            if dice <= self.__population_age_groups_info[
            self.age_group]['mortality_probability']:

                # The agent died
                dead = True

                # Agent died by natural reasons
                self.live_state = 'natural death'

        return dead


    def update_infection_status(
        self,
        becomes_into_disease_state: str
        ):
        """
        """
        if (not self.__dynamics_of_disease_states_contagion[self.disease_state]['is_infected']
        and self.__dynamics_of_disease_states_contagion[becomes_into_disease_state]['is_infected']):
            self.is_infected = True
            self.times_infected += 1
        elif (self.__dynamics_of_disease_states_contagion[self.disease_state]['is_infected']
        and not self.__dynamics_of_disease_states_contagion[becomes_into_disease_state]['is_infected']):
            self.is_infected = False

            # Inmunization level
            self.update_inmunization_level(becomes_into_disease_state)


    def update_inmunization_level(
        self,
        becomes_into_disease_state: str
        ):
        """
        """
        if self.__inmunization_level_gained_by_disease_by_vulnerability_group[
        self.vulnerability_group][becomes_into_disease_state]:
            self.inmunization_level += \
                self.__inmunization_level_gained_by_disease_by_vulnerability_group[
                    self.vulnerability_group][becomes_into_disease_state]


    def update_diagnosis_state(
        self,
        dt: float,
        previous_disease_state: str
        ):
        """
        """
        previous_diagnosis_state = self.diagnosed

        # Here we are trying to model the probability that any agent is
        # taken into account for diagnosis
        # according to its state and vulnerability group

        #=============================================
        # Not diagnosed and not waiting diagnosis ?
        # Then verify if is going to be diagnosed

        if self.__diagnosis_of_disease_states_by_vulnerability_group[
        self.vulnerability_group][self.disease_state]['can_be_diagnosed']:

            if not self.diagnosed:

                if not self.waiting_diagnosis:

                    # Verify: is going to be diagnosed ? ... Throw the dice
                    dice = np.random.random_sample()

                    if dice <= \
                    self.__diagnosis_of_disease_states_by_vulnerability_group[
                    self.vulnerability_group][self.disease_state]['diagnosis_probability']:

                        # Agent is going to be diagnosed !!!
                        self.waiting_diagnosis = True

                        self.time_waiting_diagnosis = 0

                        self.__max_time_waiting_diagnosis = \
                            self.__diagnosis_of_disease_states_by_vulnerability_group[
                                self.vulnerability_group][self.disease_state]['diagnosis_time']

                        self.__quarantine_after_being_diagnosed_enabled = \
                            self.__diagnosis_of_disease_states_by_vulnerability_group[
                                self.vulnerability_group][self.disease_state]['quarantine_after_being_diagnosed_enabled']

                        self.__quarantine_time_after_being_diagnosed = \
                            self.__diagnosis_of_disease_states_by_vulnerability_group[
                                self.vulnerability_group][self.disease_state]['quarantine_time_after_being_diagnosed']


                else:
                    # Agent is waiting diagnosis

                    #=============================================
                    # Increment time waiting diagnosis
                    self.time_waiting_diagnosis += dt

                    if self.time_waiting_diagnosis >= \
                    self.__max_time_waiting_diagnosis:

                        self.diagnosed = True

                        self.waiting_diagnosis = False

                        self.time_waiting_diagnosis = None

                        delattr(self, '_Agent__max_time_waiting_diagnosis')

                        self.time_after_being_diagnosed = 0

                        if self.__quarantine_after_being_diagnosed_enabled:
                            self.quarantine_by_diagnosis(previous_diagnosis_state)

            else:
                # Agent is diagnosed

                self.time_after_being_diagnosed += dt

                if self.__quarantine_after_being_diagnosed_enabled:

                    if self.time_after_being_diagnosed >= \
                    self.__quarantine_time_after_being_diagnosed:

                        if not self.waiting_diagnosis:

                            # Verify: is going to be diagnosed again? ... Throw the dice
                            dice = np.random.random_sample()

                            if dice <= \
                            self.__diagnosis_of_disease_states_by_vulnerability_group[
                            self.vulnerability_group][self.disease_state]['diagnosis_probability']:

                                # Agent is going to be diagnosed !!!
                                self.waiting_diagnosis = True

                                self.time_waiting_diagnosis = 0

                                self.__max_time_waiting_diagnosis = \
                                    self.__diagnosis_of_disease_states_by_vulnerability_group[
                                        self.vulnerability_group][self.disease_state]['diagnosis_time']

                                self.__quarantine_after_being_diagnosed_enabled = \
                                    self.__diagnosis_of_disease_states_by_vulnerability_group[
                                        self.vulnerability_group][self.disease_state]['quarantine_after_being_diagnosed_enabled']

                                self.__quarantine_time_after_being_diagnosed = \
                                    self.__diagnosis_of_disease_states_by_vulnerability_group[
                                        self.vulnerability_group][self.disease_state]['quarantine_time_after_being_diagnosed']

                            else:
                                # Agent was not diagnosed again !!!
                                # Agent is a false negative !!!
                                self.diagnosed = False

                                self.waiting_diagnosis = False

                                self.time_waiting_diagnosis = None

                                # self.__quarantine_after_being_diagnosed_enabled == True
                                self.quarantine_by_diagnosis(previous_diagnosis_state)

                                delattr(self, '_Agent__quarantine_after_being_diagnosed_enabled')

                                delattr(self, '_Agent__quarantine_time_after_being_diagnosed')

                        else:
                            # Agent is waiting diagnosis

                            #=============================================
                            # Increment time waiting diagnosis
                            self.time_waiting_diagnosis += dt

                            if self.time_waiting_diagnosis >= \
                            self.__max_time_waiting_diagnosis:

                                self.diagnosed = True

                                self.waiting_diagnosis = False

                                self.time_waiting_diagnosis = None

                                delattr(self, '_Agent__max_time_waiting_diagnosis')

                                self.time_after_being_diagnosed = 0

                                # self.__quarantine_after_being_diagnosed_enabled == True
                                self.quarantine_by_diagnosis(previous_diagnosis_state)

                else:
                    # self.__quarantine_after_being_diagnosed_enabled == False
                    pass
        else:
            # Agent current disease state cannot be diagnosed

            if self.waiting_diagnosis:
                # Agent previous disease state was going to be diagnosed

                #=============================================
                # Increment time waiting diagnosis
                self.time_waiting_diagnosis += dt

                if self.time_waiting_diagnosis >= \
                self.__max_time_waiting_diagnosis:

                    self.diagnosed = True

                    self.waiting_diagnosis = False

                    self.time_waiting_diagnosis = None

                    delattr(self, '_Agent__max_time_waiting_diagnosis')

                    self.time_after_being_diagnosed = 0
                    
                    if self.__quarantine_after_being_diagnosed_enabled:
                        self.quarantine_by_diagnosis(previous_diagnosis_state)

            elif self.diagnosed:
                # Agent is diagnosed

                self.time_after_being_diagnosed += dt

                if self.__quarantine_after_being_diagnosed_enabled:

                    if self.time_after_being_diagnosed >= \
                    self.__quarantine_time_after_being_diagnosed:

                        self.time_after_being_diagnosed = None

                        self.diagnosed = False

                        self.waiting_diagnosis = False

                        self.time_waiting_diagnosis = None

                        # self.__quarantine_after_being_diagnosed_enabled == True
                        self.quarantine_by_diagnosis(previous_diagnosis_state)

                        delattr(self, '_Agent__quarantine_after_being_diagnosed_enabled')

                        delattr(self, '_Agent__quarantine_time_after_being_diagnosed')

                else:
                    # self.__quarantine_after_being_diagnosed_enabled == False
                    pass

            else:
                self.diagnosed = False

                self.waiting_diagnosis = False

                self.time_waiting_diagnosis = None

                self.quarantine_by_diagnosis(previous_diagnosis_state)


    def update_hospitalization_state(
        self,
        dt: int,
        hospitals: dict,
        hospitals_spatial_tree,
        hospitals_labels
        ):
        """
        """
        if self.disease_state == 'dead' and self.is_hospitalized:
            hospitals[self.hospital_label].remove_hospilized_agent(self.agent)
            if self.is_in_UCI:
                hospitals[self.hospital_label].remove_agent_from_UCI(self.agent)
                return hospitals
            else:
                return hospitals

        previous_hospitalization_state = self.is_hospitalized
        previous_UCI_state = self.is_in_UCI

        # Retrieve agent location
        agent_location = [self.x, self.y]

        # Agent should be hospitalized ?
        if self.__hospitalization_of_disease_states_by_vulnerability_group[
        self.vulnerability_group][self.disease_state]['should_be_hospitalized']:

            #=============================================
            if not self.needs_hospitalization:

                #=============================================
                # Check if needs hospitalization ... Throw the dice
                dice = np.random.random_sample()

                if dice <= \
                self.__hospitalization_of_disease_states_by_vulnerability_group[
                self.vulnerability_group][self.disease_state]['hospitalization_probability']:

                    # Agent needs hospitalization !!!
                    self.needs_hospitalization = True
                    self.time_waiting_hospitalization = 0

            #=============================================
            if self.needs_hospitalization and not self.is_hospitalized:

                nearest_hospital_index = \
                    hospitals_spatial_tree.query(agent_location)[1]

                nearest_hospital_label = hospitals_labels[nearest_hospital_index]

                agent_was_added = hospitals[nearest_hospital_label].add_hospilized_agent(self.agent)

                if agent_was_added:
                    self.is_hospitalized = True
                    self.hospital_label = nearest_hospital_label
                    self.needs_hospitalization = False
                    self.time_waiting_hospitalization = None

                    [self.x, self.y] = hospitals[self.hospital_label].get_hospital_location()
                    self.vx = 0.
                    self.vy = 0.
                else:
                    self.is_hospitalized = False
                    self.time_waiting_hospitalization += dt

                    # Disease transition
                    self.disease_states_transitions_by_hospitalization_unavailability()

            #=============================================
            if (self.is_hospitalized
            and self.__hospitalization_of_disease_states_by_vulnerability_group[
            self.vulnerability_group][self.disease_state]['UCI_probability'] is not 0.0):

                if not self.needs_UCI and not self.is_in_UCI:
                    #=============================================
                    # Verify: needs UCI ? ... Throw the dice
                    dice = np.random.random_sample()

                    if dice <= \
                    self.__hospitalization_of_disease_states_by_vulnerability_group[
                    self.vulnerability_group][self.disease_state]['UCI_probability']:

                        # Agent needs UCI !!!
                        self.needs_UCI = True
                        self.time_waiting_UCI = 0

                #=============================================
                if self.needs_UCI and not self.is_in_UCI:

                    agent_was_added = hospitals[self.hospital_label].add_agent_to_UCI(self.agent)

                    if agent_was_added:
                        self.is_in_UCI = True
                        self.needs_UCI = False
                        self.time_waiting_UCI = None
                    else:
                        self.is_in_UCI = False
                        self.time_waiting_UCI += dt

                        # Disease transition
                        self.disease_states_transitions_by_UCI_unavailability()

            #=============================================
            if (self.is_hospitalized
            and self.__hospitalization_of_disease_states_by_vulnerability_group[
            self.vulnerability_group][self.disease_state]['UCI_probability'] == 0.0):

                # Agen should not be in UCI but should remain hospitalized
                self.needs_UCI = False
                self.is_in_UCI = False
                self.time_waiting_UCI = None

                if previous_UCI_state:
                    hospitals[self.hospital_label].remove_agent_from_UCI(self.agent)

        else:
            # Agent should not be hospitalized
            self.needs_hospitalization = False
            self.is_hospitalized = False
            self.time_waiting_hospitalization = None

            if previous_hospitalization_state:
                hospitals[self.hospital_label].remove_hospilized_agent(self.agent)

                # TODO
                # Change this positions in order to use random
                # if hospital location is None
                if self.x == None:
                    self.x = 0.
                if self.y == None:
                    self.y = 0.

                # Define velocity between (0 , vmax)
                self.initialize_velocity()

            self.needs_UCI = False
            self.is_in_UCI = False
            self.time_waiting_UCI = None

            if previous_UCI_state:
                hospitals[self.hospital_label].remove_agent_from_UCI(self.agent)

            self.hospital_label = None

        return hospitals


    def trace_neighbors(
        self,
        spatial_trees_by_disease_state: dict,
        agents_indices_by_disease_state: dict
        ):
        """
        """
        # Initialize
        self.susceptible_neighbors = []
        self.infected_spreader_neighbors = np.array([])
        self.infected_non_spreader_neighbors = np.array([])
        self.inmune_neighbors = []
        self.total_neighbors = []

        # Retrieve agent location
        agent_location = [self.x, self.y]

        # Excluded states for neighbors search
        excluded_states = ['death']

        # Cycle through each state of the neighbors
        for disease_state in set(self.__disease_states) - set(excluded_states):

            if spatial_trees_by_disease_state[disease_state]:

                # Detect if any agent in "disease_state" is inside a distance
                # equal to the corresponding spread_radius
                points_inside_radius = \
                    spatial_trees_by_disease_state[disease_state].query_ball_point(
                        agent_location,
                        self.tracing_radius
                        )

                agents_indices_inside_radius = agents_indices_by_disease_state[
                    disease_state][points_inside_radius]

                agents_indices_inside_radius = np.setdiff1d(
                    agents_indices_inside_radius,
                    self.agent
                    )

                if disease_state == 'susceptible':
                    self.susceptible_neighbors = agents_indices_inside_radius.tolist()

                elif disease_state == 'inmune':
                    self.inmune_neighbors = agents_indices_inside_radius.tolist()

                elif self.__dynamics_of_disease_states_contagion[
                disease_state]['is_infected']:
                    # If not susceptible and not inmune
                    # then it must be infected

                    if self.__dynamics_of_disease_states_contagion[
                    disease_state]['can_spread']:

                        if len(agents_indices_inside_radius) is not 0:

                            self.infected_spreader_neighbors = np.concatenate(
                                (self.infected_spreader_neighbors, agents_indices_inside_radius),
                                axis=None
                                )

                    else:
                        # self.__dynamics_of_disease_states_contagion[disease_state]['can_spread'] == False
                        if len(agents_indices_inside_radius) is not 0:

                            self.infected_non_spreader_neighbors = np.concatenate(
                                (self.infected_non_spreader_neighbors, agents_indices_inside_radius),
                                axis=None
                                )

                else:
                    # print error ?
                    pass

        # float to int
        self.infected_spreader_neighbors = self.infected_spreader_neighbors.astype(int)
        self.infected_non_spreader_neighbors = self.infected_non_spreader_neighbors.astype(int)

        # ndarray to list
        self.infected_spreader_neighbors = self.infected_spreader_neighbors.tolist()
        self.infected_non_spreader_neighbors = self.infected_non_spreader_neighbors.tolist()

        # Extend total_neighbors
        self.total_neighbors.extend(self.susceptible_neighbors)
        self.total_neighbors.extend(self.inmune_neighbors)
        self.total_neighbors.extend(self.infected_spreader_neighbors)
        self.total_neighbors.extend(self.infected_non_spreader_neighbors)


    def update_alertness_state(
        self,
        spatial_trees_by_disease_state: dict,
        agents_indices_by_disease_state: dict,
        population_positions: dict,
        population_velocities: dict
        ):
        """
        """
        # Initialize
        self.avoidable_neighbors = np.array([])
        self.alertness = False
        self.alerted_by = []

        # Retrieve agent location
        agent_location = [self.x, self.y]

        #=============================================
        # Agent should be alert?
        if self.__dynamics_of_alertness_of_disease_states_by_vulnerability_group[
            self.vulnerability_group][self.disease_state]['should_be_alert']:

            # Initialize "alerted by"
            alerted_by = []

            # Cycle through each state of the neighbors to see if the agent
            # should be alert
            for disease_state in self.__disease_states:

                # Note that an 'avoidable_agent' depends on its state but
                # also on each group, it means that each group defines
                # which state is avoidable
                if (self.__dynamics_of_avoidance_of_disease_states_by_vulnerability_group[
                    self.vulnerability_group][disease_state]['avoidable_agent']
                    and spatial_trees_by_disease_state[disease_state]):

                    # Detect if any avoidable agent is inside a distance
                    # equal to the corresponding avoidness_radius
                    points_inside_radius = \
                        spatial_trees_by_disease_state[disease_state].query_ball_point(
                            agent_location,
                            self.__dynamics_of_avoidance_of_disease_states_by_vulnerability_group[
                            self.vulnerability_group][disease_state]['avoidness_radius']
                            )

                    avoidable_neighbors = \
                        agents_indices_by_disease_state[disease_state][points_inside_radius]

                    # If agent_index in avoidable_neighbors
                    # then remove it
                    if self.agent in avoidable_neighbors:

                        avoidable_neighbors = np.setdiff1d(
                            avoidable_neighbors,
                            self.agent
                            )

                    if len(avoidable_neighbors) is not 0:

                        self.avoidable_neighbors = np.concatenate(
                            (self.avoidable_neighbors, avoidable_neighbors),
                            axis=None
                            )

                        for avoidable_agent_index in avoidable_neighbors:

                            # Must agent be alert ? ... Throw the dice
                            dice = np.random.random_sample()

                            # Note that alertness depends on a probability,
                            # which tries to model the probability that an
                            # agent with a defined group and state is alert
                            if (dice <= \
                                self.__dynamics_of_alertness_of_disease_states_by_vulnerability_group[
                                self.vulnerability_group][self.disease_state]['alertness_probability']
                                ):

                                # Agent is alerted !!!
                                self.alertness = True

                                # Append avoidable_agent_index in alerted_by
                                alerted_by.append(avoidable_agent_index)

            # float to int
            self.avoidable_neighbors = self.avoidable_neighbors.astype(int)

            # ndarray to list
            self.avoidable_neighbors = self.avoidable_neighbors.tolist()

            #=============================================
            # Change movement direction if agent's alertness = True
            if self.alertness:

                # Append alerted_by array to self.alerted_by 
                self.alerted_by = alerted_by

                # Retrieve positions of the agents to be avoided
                positions_to_avoid = [
                    population_positions[agent_to_be_avoided]
                    for agent_to_be_avoided in self.alerted_by
                    ]

                # Retrieve velocities of the agents to be avoided
                velocities_to_avoid = [
                    population_velocities[agent_to_be_avoided]
                    for agent_to_be_avoided in self.alerted_by
                    ]

                # Change movement direction
                self.alert_avoid_agents(
                    positions_to_avoid,
                    velocities_to_avoid
                    )


    def alert_avoid_agents(
        self,
        positions_to_avoid: list,
        velocities_to_avoid: list
        ):
        """
        """
        if (not self.is_hospitalized
        and self.quarantine_state == 'Not in quarantine'):

            #=============================================
            # Starting parameters

            # Avoid numerical issues when cosine tends to 1
            epsilon = 0.002

            # 0.349066  = 20° you can set the range of final random angular deviation
            random_rotation_angle_range = 0.349066

            random_rotation_angle = np.random.uniform(
                -random_rotation_angle_range,
                random_rotation_angle_range
                )

            velocity_changed = False

            #=============================================
            # Retrieve own velocity and position

            own_initial_position = np.array([self.x, self.y])

            own_initial_velocity = np.array([self.vx, self.vy])

            norm_own_initial_velocity = np.sqrt(
                np.dot(own_initial_velocity, own_initial_velocity)
                )

            #=============================================
            # Create a vector to save new velocity vector
            new_velocity = np.array([0., 0.], dtype='float64')

            #=============================================
            # Avoid velocities and positions
            for (position_to_avoid, velocity_to_avoid) in zip(
                positions_to_avoid,
                velocities_to_avoid
                ):

                # Locate the avoidable agents position
                relative_position_vector = position_to_avoid - own_initial_position

                relative_position_vector_norm = np.sqrt(
                    np.dot(relative_position_vector, relative_position_vector)
                    )

                velocity_to_avoid_norm = np.sqrt(
                    np.dot(velocity_to_avoid, velocity_to_avoid)
                    )

                # alpha: angle between agent velocity and
                #        relative_position_vector of avoidable agent
                # beta: angle between - relative_position
                #       and avoidable agent velocity
                # theta: angle between agents velocities

                cosalpha = \
                    np.dot(relative_position_vector, own_initial_velocity) \
                        / (relative_position_vector_norm * norm_own_initial_velocity)

                cosalpha = 1.0 \
                    if cosalpha > 1.0 else (-1.0 if cosalpha < -1.0 else cosalpha)

                if velocity_to_avoid_norm != 0.:
                    # If avoidable agent isn't at rest
                    costheta = \
                        np.dot(own_initial_velocity, velocity_to_avoid) \
                            / (norm_own_initial_velocity * velocity_to_avoid_norm)

                    costheta = 1.0 \
                        if costheta > 1.0 else (-1.0 if costheta < -1.0 else costheta)

                    cosbeta = \
                        np.dot(-relative_position_vector, velocity_to_avoid) \
                            / (relative_position_vector_norm * velocity_to_avoid_norm)
                    cosbeta = 1.0 \
                        if cosbeta > 1.0 else (-1.0 if cosbeta < -1.0 else cosbeta)

                    if (cosbeta <= (1.0 - epsilon)
                        and cosbeta >= (-1.0 + epsilon)
                        ):
                        # if beta is not 0 or pi
                        # define the perpendicular direction to
                        # the avoidable agent velocity
                        new_direction = \
                            np.dot(relative_position_vector, velocity_to_avoid) \
                            * velocity_to_avoid \
                            - velocity_to_avoid_norm**2 * relative_position_vector

                        new_direction_normalized = \
                            new_direction \
                                / np.sqrt(np.dot(new_direction, new_direction))

                    else:
                        new_direction_normalized = \
                            np.array([velocity_to_avoid[1], - velocity_to_avoid[0]])\
                                / velocity_to_avoid_norm

                    # Including a random deviation
                    new_direction_rotated = np.array([(new_direction_normalized[0]*np.cos(random_rotation_angle) -
                                                    new_direction_normalized[1]*np.sin(random_rotation_angle))
                                                    ,(new_direction_normalized[0]*np.sin(random_rotation_angle) + new_direction_normalized[1]*np.cos(random_rotation_angle))])

                    if (-1. <= cosalpha <= 0.):
                        # Avoidable agent is at agents back, pi<alpha<3pi/2
                        # If avoidable agent can reach agent trajectory,
                        # agent changes its direction
                        if velocity_to_avoid_norm >= norm_own_initial_velocity:

                            if (cosbeta >= -cosalpha and costheta >= -cosalpha):
                                new_velocity += norm_own_initial_velocity * new_direction_rotated
                                velocity_changed = True
                    
                            else: # If both agents can't intersect
                                new_velocity += np.array([0.,0.], dtype='float64')

                        else:  # If infected velocity < agent, don't worry
                            new_velocity += np.array([0.,0.], dtype='float64')

                    if (0. < cosalpha <= 1.):
                        # Looking foward: Avoid everyone
                        new_velocity += norm_own_initial_velocity * new_direction_rotated
                        velocity_changed = True
                
                # If avoidable agent is at rest and in forwards,
                # avoid it changing the direction towards -relative_position_vector
                # with a random deviation 
                else: 
                    if (0. < cosalpha < 1.):
                        new_velocity += - np.array([(relative_position_vector[0]*np.cos(random_rotation_angle) - relative_position_vector[1]*np.sin(random_rotation_angle))
                                                ,(relative_position_vector[0]*np.sin(random_rotation_angle) + relative_position_vector[1]*np.cos(random_rotation_angle))])*norm_own_initial_velocity/relative_position_vector_norm 
                        velocity_changed = True 
                    else:
                        new_velocity += np.array([0.,0.], dtype='float64')

            # Disaggregate new_velocity
            if not np.array_equal(new_velocity, np.array([0.,0.], dtype='float64')):
                self.vx, self.vy = new_velocity

            elif np.array_equal(new_velocity, np.array([0.,0.], dtype='float64')) \
                and velocity_changed:
                self.vx, self.vy = new_velocity

            else:
                # Velocity doesn't change
                pass


    def move(
        self,
        dt: float,
        maximum_free_random_speed_factor: float,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float
        ):
        """
        """
        if self.quarantine_state == 'Not in quarantine':

            #=============================================
            # Avoid frontier
            self.avoid_frontier(dt, xmin, xmax, ymin, ymax)

            #=============================================
            # Physical movement
            self.physical_movement(dt, maximum_free_random_speed_factor)


    def avoid_frontier(
        self,
        dt: float,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float
        ):
        """
        """
        #=============================================
        # Bouncing from the walls
        new_x = self.x + self.vx * dt
        new_y = self.y + self.vy * dt

        #=============================================
        # Hitting upper or lower wall 
        if new_x > xmax or new_x < xmin:
            # self.x = self.x - self.vx * dt
            self.vx = - self.vx

        #=============================================
        # Hitting left or right wall
        if new_y > ymax or new_y < ymin:
            # self.y = self.y - self.vy * dt
            self.vy = - self.vy


    def physical_movement(
        self,
        dt: float,
        maximum_free_random_speed_factor: float
        ):
        """
        """
        if not self.is_hospitalized:
            #=============================================
            # Evolve position
            self.x = self.x + self.vx * dt
            self.y = self.y + self.vy * dt


            #=============================================
            # Evolve velocity as a random walk

            v_squared_norm = self.vx**2 + self.vy**2

            v_norm = np.sqrt(v_squared_norm)

            # sin_theta = self.vy / v_norm if self.vy < v_norm else 1.0

            # current_theta = np.arcsin(sin_theta)

            random_factor = 2 * np.random.random_sample() - 1 # [-1, 1)

            dv = maximum_free_random_speed_factor * v_norm * random_factor

            # The angle can be whatever between [0, 2 pi)
            dtheta = 2. * np.pi * np.random.random_sample()

            dvx, dvy = [dv * np.cos(dtheta), dv * np.sin(dtheta)]

            new_vx = self.vx + dvx
            new_vy = self.vy + dvy

            new_v_norm = np.sqrt(new_vx**2 + new_vy**2)

            if new_v_norm <= self.vmax:
                self.vx, self.vy = [new_vx, new_vy]
            else:
                sin_new_theta = new_vy / new_v_norm if new_vy < new_v_norm else 1.0

                new_theta = np.arcsin(sin_new_theta)

                self.vx, self.vy = [
                    self.vmax * np.cos(new_theta),
                    self.vmax * np.sin(new_theta)
                    ]


    def quarantine_by_diagnosis(
        self,
        previous_diagnosis_state: bool
        ):
        if previous_diagnosis_state and self.diagnosed:
            pass
        elif not previous_diagnosis_state and self.diagnosed:

            # Is agent obedient ? ... Throw the dice
            dice = np.random.random_sample()

            if dice <= self.obedience_to_quarantine:
                self.quarantine_state = 'Quarantine by diagnosis'
                self.vx = 0.
                self.vy = 0.
            else:
                self.quarantine_state = 'Quarantine by diagnosis - Disobedient'

        elif previous_diagnosis_state and not self.diagnosed:

            self.quarantine_state = 'Not in quarantine'

            # Define velocity between (0 , vmax)
            self.initialize_velocity()

        else:
            # not previous_diagnosis_state and not self.diagnosed
            pass


    def quarantine_by_government_decrees(
        self,
        decreed_quarantine: bool,
        groups_in_quarantine: list
        ):
        """
        """
        if self.quarantine_state not in {
            'Quarantine by diagnosis', 
            'Quarantine by diagnosis - Disobedient'
            }:

            if decreed_quarantine:

                if self.quarantine_group in groups_in_quarantine:

                    if self.quarantine_state == 'Not in quarantine':

                        # Is agent obedient ? ... Throw the dice
                        dice = np.random.random_sample()

                        if dice <= self.obedience_to_quarantine:
                            self.quarantine_state = 'Quarantine by government'
                            self.vx = 0.
                            self.vy = 0.
                        else:
                            self.quarantine_state = 'Quarantine by government - Disobedient'

                else:
                    if self.quarantine_state == 'Quarantine by government':

                        self.quarantine_state = 'Not in quarantine'

                        # Define velocity between (0 , vmax)
                        self.initialize_velocity()

                    if self.quarantine_state == 'Quarantine by government - Disobedient':
                        self.quarantine_state = 'Not in quarantine'

            else:
                if self.quarantine_state != 'Not in quarantine':
                    self.quarantine_state = 'Not in quarantine'




