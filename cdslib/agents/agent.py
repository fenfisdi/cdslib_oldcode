import copy
import numpy as np

dim = 2

class AgentsInfo:
    """
    """
    def __init__(
        self,
        disease_states: list,
        vulnerability_groups: list,
        dynamics_of_the_disease_states_transitions_info: dict,
        population_age_groups_info: dict,
        mortality_of_disease_states_by_vulnerability_group: dict,
        diagnosis_of_disease_states_by_vulnerability_group: dict,
        hospitalization_of_disease_states_by_vulnerability_group: dict
    ):
        """
        """
        self.disease_states = disease_states
        self.vulnerability_groups = vulnerability_groups

        # dynamics_of_the_disease_states_transitions_info
        self.disease_states_time_functions = \
            dynamics_of_the_disease_states_transitions_info[
                'disease_states_time_functions'
                ]

        self.disease_states_transitions_by_vulnerability_group = \
            dynamics_of_the_disease_states_transitions_info[
                'disease_states_transitions_by_vulnerability_group'
                ]

        self.population_age_groups_info = population_age_groups_info
        self.mortality_of_disease_states_by_vulnerability_group = \
            mortality_of_disease_states_by_vulnerability_group
        self.diagnosis_of_disease_states_by_vulnerability_group = \
            diagnosis_of_disease_states_by_vulnerability_group
        self.hospitalization_of_disease_states_by_vulnerability_group = \
            hospitalization_of_disease_states_by_vulnerability_group


class Agent:
    """
    """
    def __init__(
        self,
        agents_info: AgentsInfo,
        agent: int,
        x: float,
        y: float,
        vmax: float,
        group: str,
        state: str,
        age: float,
        diagnosed: bool=False,
        inmunization_level: float=0.0
    ):
        """
        """
        # Define private atributes from AgentsInfo atributes
        self.__disease_states = agents_info.disease_states
        self.__vulnerability_groups = agents_info.vulnerability_groups
        self.__disease_states_time_functions = \
            agents_info.disease_states_time_functions
        self.__disease_states_transitions_by_vulnerability_group = \
            agents_info.disease_states_transitions_by_vulnerability_group
        self.__population_age_groups_info = \
            agents_info.population_age_groups_info
        self.__mortality_of_disease_states_by_vulnerability_group = \
            agents_info.mortality_of_disease_states_by_vulnerability_group
        self.__diagnosis_of_disease_states_by_vulnerability_group = \
            agents_info.diagnosis_of_disease_states_by_vulnerability_group
        self.__hospitalization_of_disease_states_by_vulnerability_group = \
            agents_info.hospitalization_of_disease_states_by_vulnerability_group

        # Define agent label
        self.agent = agent

        # Define position
        self.x = x
        self.y = y

        # Define initial velocity between (-vmax , +vmax)
        self.vx, self.vy = 2. * vmax * np.random.random_sample(dim) - vmax

        # Initialize group, state, and diagnosed state
        self.group = group

        self.state = state

        self.diagnosed = diagnosed

        self.hospitalized = False

        self.requires_UCI = False

        self.waiting_diagnosis = False

        self.time_waiting_diagnosis = None

        self.infected_by = None

        self.inmunization_level = inmunization_level

        self.alertness = False

        self.alerted_by = []

        self.state_time = 0

        self.live_state = 'alive'

        self.age = age

        # Determine age group
        self.determine_age_group()

        # Determine state time
        self.determine_state_time()


    def getstate(self):

        agent_dict = copy.deepcopy(self.__dict__)

        # Remove private attributes
        del agent_dict['_Agent__disease_states']
        del agent_dict['_Agent__vulnerability_groups']
        del agent_dict['_Agent__disease_states_time_functions']
        del agent_dict['_Agent__disease_states_transitions_by_vulnerability_group']
        del agent_dict['_Agent__population_age_groups_info']
        del agent_dict['_Agent__mortality_of_disease_states_by_vulnerability_group']
        del agent_dict['_Agent__diagnosis_of_disease_states_by_vulnerability_group']
        del agent_dict['_Agent__hospitalization_of_disease_states_by_vulnerability_group']

        return agent_dict


    def getkeys(self):

        agent_dict = self.getstate()

        return agent_dict.keys()


    def determine_state_time(self):
        """
        """
        if self.__disease_states_time_functions[
            self.group][self.state]['time_function']:

            self.state_max_time = \
                self.__disease_states_time_functions[
                self.group][self.state]['time_function']()

        else:
            self.state_max_time = None


    def state_transition(
        self,
        dt: float
        ):
        """
        """
        self.state_time += dt

        if (self.state_max_time and self.state_time == self.state_max_time):

            # Verify: becomes into ? ... Throw the dice
            dice = np.random.random_sample()

            cummulative_probability = 0

            # OJO con SORTED
            for (probability, becomes_into_state) in zip(
                self.__disease_states_transitions_by_vulnerability_group[
                    self.group][self.state]['transition_probability'],
                self.__disease_states_transitions_by_vulnerability_group[
                    self.group][self.state]['becomes_into']
                ):
                cummulative_probability += probability

                if dice <= cummulative_probability:
                    
                    self.state = becomes_into_state
                    
                    self.update_diagnosis_state(dt, the_state_changed=True)
                    self.update_hospitalization_state()

                    self.state_time = 0

                    self.determine_state_time()

                    break
        
        self.update_diagnosis_state(dt, the_state_changed=False)
        self.update_hospitalization_state()


    def determine_age_group(self):
        """
        """
        for age_group in self.__population_age_groups_info.keys():

            age_group_dict = self.__population_age_groups_info[age_group]

            if (age_group_dict['min_age'] <= self.age
            and np.floor(self.age) <= age_group_dict['max_age']):

                self.age_group = age_group


    def age_and_death_verification(
        self,
        dt_scale_in_years: float
        ):
        """
        """
        dead = False

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

        #=============================================
        # Verify: dead by disease ? ... Throw the dice
        dice = np.random.random_sample()

        if self.__mortality_of_disease_states_by_vulnerability_group[
            self.group][self.state]['mortality_probability']:
            if dice <= self.__mortality_of_disease_states_by_vulnerability_group[
                self.group][self.state]['mortality_probability']:
                # The agent died
                dead = True

                # Agent die by disease
                self.live_state = 'dead by disease'

        return dead


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
                        self.group][self.state]['can_be_diagnosed']:

                        # Verify: is going to be diagnosed ? ... Throw the dice
                        dice = np.random.random_sample()

                        if dice <= self.__diagnosis_of_disease_states_by_vulnerability_group[
                            self.group][self.state]['diagnosis_probability']:

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
                        self.group][self.state]['diagnosis_time']:

                        # Can be diagnosed ? (in other words: Is infected ?)
                        if self.__diagnosis_of_disease_states_by_vulnerability_group[
                            self.group][self.state]['can_be_diagnosed']:

                            self.diagnosed = True

                            self.waiting_diagnosis = False

                            self.time_waiting_diagnosis = None

                        else:
                            self.diagnosed = False

                            self.waiting_diagnosis = False

                            self.time_waiting_diagnosis = None

            else:
                # Agent is diagnosed
                pass
        else:
            # Agent changed state

            # If agent cannot be diagnosed
            if not self.__diagnosis_of_disease_states_by_vulnerability_group[
                self.group][self.state]['can_be_diagnosed']:
                
                self.diagnosed = False

                self.waiting_diagnosis = False

                self.time_waiting_diagnosis = None


    def update_hospitalization_state(self):
        # Agent can be hospitalized ?
        if self.__hospitalization_of_disease_states_by_vulnerability_group[
            self.group][self.state]['can_be_hospitalized']:
        
            if not self.hospitalized:

                #=============================================
                # Verify: is hospitalized ? ... Throw the dice
                dice = np.random.random_sample()

                if (
                self.__hospitalization_of_disease_states_by_vulnerability_group[
                    self.group][self.state]['hospitalization_probability']
                and dice <= \
                self.__hospitalization_of_disease_states_by_vulnerability_group[
                    self.group][self.state]['hospitalization_probability']):
                    
                    # Agent is hospitalized !!!
                    self.hospitalized = True

                    #=============================================
                    # Verify: needs UCI ? ... Throw the dice
                    dice = np.random.random_sample()

                    if (
                        self.__hospitalization_of_disease_states_by_vulnerability_group[
                            self.group][self.state]['UCI_probability']
                        and dice <= \
                        self.__hospitalization_of_disease_states_by_vulnerability_group[
                            self.group][self.state]['UCI_probability']
                        ):
                        
                        # Agent needs UCI !!!
                        self.requires_UCI = True

            else:
                pass
        else:
            # Agent can not be hospitalized ?
            self.hospitalized = False

            self.requires_UCI = False


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
        self.physical_movement(dt,  maximum_free_random_speed)


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
