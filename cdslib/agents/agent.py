import copy
import numpy as np

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
        hospitalization_info: dict,
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

        self.inmunization_level_by_vulnerability_group = \
            contagion_dynamics_info[
                'inmunization_level_by_vulnerability_group'
                ]

        # population_age_groups_info
        self.population_age_groups_info = population_age_groups_info

        self.diagnosis_of_disease_states_by_vulnerability_group = \
            diagnosis_of_disease_states_by_vulnerability_group

        # hospitalization_info
        self.hospital_information = \
            hospitalization_info[
                    'hospital_information'
                ]

        self.hospitalization_of_disease_states_by_vulnerability_group = \
            hospitalization_info[
                    'hospitalization_of_disease_states_by_vulnerability_group'
                ]

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
        diagnosed: bool=False,
        inmunization_level: float=0.0
        ):
        """
        """
        # Define private atributes from AgentsInfo atributes
        for key, value in zip(
            agents_info.__dict__.keys(),
            agents_info.__dict__.values()
            ):
            setattr(self, '_Agent__' + key, value)

        # Define agent label
        self.agent: int = agent

        self.age: float = age

        self.age_group: str = age_group

        # Define position
        self.x: float = x
        self.y: float = y

        # Define initial velocity between (-vmax , +vmax)
        self.vx, self.vy = 2. * vmax * np.random.random_sample(dim) - vmax

        # Initialize disease_state, susceptibility_group,
        # vulnerability_group and diagnosis state 
        self.disease_state: str = disease_state

        self.susceptibility_group: str = susceptibility_group

        self.vulnerability_group: str = vulnerability_group

        self.diagnosed: bool = diagnosed

        self.hospitalized: bool = False

        self.is_in_UCI: bool = False

        self.waiting_diagnosis: bool = False

        self.time_waiting_diagnosis: int = None

        self.infected_by: list = []
        self.infected_in_step: int = None
        self.infected_info: dict = {}

        self.inmunization_level: float = inmunization_level

        self.contacted_with: list = []

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
        dt: float
        ):
        """
        """
        self.disease_state_time += dt

        if (self.disease_state_max_time
        and self.disease_state_time == self.disease_state_max_time):

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
                        # Agent die by disease
                        self.live_state = 'dead by disease'
                    else:
                        self.update_diagnosis_state(dt, the_state_changed=True)
                        self.update_hospitalization_state()

                    self.disease_state_time = 0

                    self.determine_disease_state_time()

                    break

        self.update_diagnosis_state(dt, the_state_changed=False)
        self.update_hospitalization_state()


    def disease_state_transition_by_contagion(
        self,
        step: int,
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

                        self.disease_state = becomes_into_disease_state

                        self.update_diagnosis_state(step, the_state_changed=True)

                        self.disease_state_time = 0

                        self.determine_disease_state_time()

                        break
        else:
            # Agent cannot get infected
            pass


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
        elif not self.__dynamics_of_disease_states_contagion[becomes_into_disease_state]['is_infected']:
            self.is_infected = False

            # Inmunization level
            self.update_inmunization_level()


    def update_inmunization_level(self):
        """
        """
        self.inmunization_level += \
            self.__inmunization_level_by_vulnerability_group[self.vulnerability_group]


    def update_diagnosis_state(
        self,
        dt: float,
        the_state_changed: bool=False
        ):
        """
        """
        if not the_state_changed:
            # Here we are trying to model the probability that any agent is
            # taken into account for diagnosis
            # according to its state and vulnerability group

            #=============================================
            # Not diagnosed and not waiting diagnosis ?
            # Then verify if is going to be diagnosed

            if not self.diagnosed:

                if not self.waiting_diagnosis:

                    if self.__diagnosis_of_disease_states_by_vulnerability_group[
                    self.vulnerability_group][self.disease_state]['can_be_diagnosed']:

                        # Verify: is going to be diagnosed ? ... Throw the dice
                        dice = np.random.random_sample()

                        if dice <= \
                        self.__diagnosis_of_disease_states_by_vulnerability_group[
                        self.vulnerability_group][self.disease_state]['diagnosis_probability']:

                            # Agent is going to be diagnosed !!!
                            self.waiting_diagnosis = True

                            self.time_waiting_diagnosis = 0

                else:
                    # Agent is waiting diagnosis

                    #=============================================
                    # Increment time waiting diagnosis
                    self.time_waiting_diagnosis += dt

                    if self.time_waiting_diagnosis == \
                    self.__diagnosis_of_disease_states_by_vulnerability_group[
                    self.vulnerability_group][self.disease_state]['diagnosis_time']:

                        self.diagnosed = True

                        self.waiting_diagnosis = False

                        self.time_waiting_diagnosis = None

            else:
                # Agent is diagnosed
                pass
        else:
            # Agent changed state

            # If agent cannot be diagnosed
            if not self.__diagnosis_of_disease_states_by_vulnerability_group[
            self.vulnerability_group][self.disease_state]['can_be_diagnosed']:

                self.diagnosed = False

                self.waiting_diagnosis = False

                self.time_waiting_diagnosis = None


    def update_hospitalization_state(self):
        """
        """
        # Agent can be hospitalized ?
        if self.__hospitalization_of_disease_states_by_vulnerability_group[
        self.vulnerability_group][self.disease_state]['can_be_hospitalized']:

            if not self.hospitalized:

                #=============================================
                # Verify: is hospitalized ? ... Throw the dice
                dice = np.random.random_sample()

                if dice <= \
                self.__hospitalization_of_disease_states_by_vulnerability_group[
                self.vulnerability_group][self.disease_state]['hospitalization_probability']:

                    # Agent is hospitalized !!!
                    # TODO change to "needs_hospitalization"
                    self.hospitalized = True

                    #=============================================
                    # Verify: needs UCI ? ... Throw the dice
                    dice = np.random.random_sample()

                    if dice <= \
                    self.__hospitalization_of_disease_states_by_vulnerability_group[
                    self.vulnerability_group][self.disease_state]['UCI_probability']:

                        # Agent needs UCI !!!
                        # TODO change to "needs_UCI"
                        self.is_in_UCI = True

            else:
                # Agent is hospitalized

                if not self.is_in_UCI:

                    #=============================================
                    # Verify: needs UCI ? ... Throw the dice
                    dice = np.random.random_sample()

                    if dice <= \
                    self.__hospitalization_of_disease_states_by_vulnerability_group[
                    self.vulnerability_group][self.disease_state]['UCI_probability']:

                        # Agent needs UCI !!!
                        self.is_in_UCI = True
        else:
            # Agent can not be hospitalized ?
            self.hospitalized = False

            self.is_in_UCI = False


    def update_alertness_state(
        self,
        step: int,
        spatial_trees_by_disease_state: dict,
        agents_indices_by_disease_state: dict,
        population_positions: dict,
        population_velocities: dict
        ):
        """
        """
        # Initialize agent alertness
        self.alertness = False
        self.contacted_with = []
        self.alerted_by = []

        #=============================================
        # Agent should be alert?
        if self.__dynamics_of_alertness_of_disease_states_by_vulnerability_group[
            self.vulnerability_group][self.disease_state]['should_be_alert']:
            
            # Retrieve agent location
            agent_location = [self.x, self.y]

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
                    and spatial_trees_by_disease_state[disease_state]
                    ):
                    
                    # Detect if any avoidable agent is inside a distance
                    # equal to the corresponding spread_radius
                    points_inside_radius = \
                        spatial_trees_by_disease_state[disease_state].query_ball_point(
                            agent_location,
                            self.__dynamics_of_avoidance_of_disease_states_by_vulnerability_group[
                            self.vulnerability_group][disease_state]['avoidness_radius']
                            )

                    avoidable_indices_inside_radius = \
                        agents_indices_by_disease_state[disease_state][points_inside_radius]
                    
                    # If agent_index in avoidable_indices_inside_radius,
                    # then remove it
                    if self.agent in avoidable_indices_inside_radius:
                        
                        avoidable_indices_inside_radius = np.setdiff1d(
                            avoidable_indices_inside_radius,
                            self.agent
                            )

                    if len(avoidable_indices_inside_radius) is not 0:

                        # Append avoidable_indices_inside_radius array
                        # to self.contacted_with
                        self.contacted_with = avoidable_indices_inside_radius
                        
                        for avoidable_agent_index in avoidable_indices_inside_radius:

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
        maximum_free_random_speed: float,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float
        ):
        """
        """
        #=============================================
        # Avoid frontier
        self.avoid_frontier(dt, xmin, xmax, ymin, ymax)

        #=============================================
        # Physical movement
        self.physical_movement(dt, maximum_free_random_speed)


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
        maximum_free_random_speed: float
        ):
        """
        """
        #=============================================
        # Evolve position
        self.x = self.x + self.vx * dt
        self.y = self.y + self.vy * dt


        if not self.alertness:
            #=============================================
            # Evolve velocity as a random walk
            dvx = maximum_free_random_speed * np.random.random_sample() \
                - maximum_free_random_speed/2
            self.vx = self.vx + dvx

            dvy = maximum_free_random_speed * np.random.random_sample() \
                - maximum_free_random_speed/2
            self.vy = self.vy + dvy
