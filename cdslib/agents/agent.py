import numpy as np

dim = 2

class AgentsInfo:
    """
    """
    def __init__(
        self,
        disease_states: list,
        vulnerability_groups: list,
        population_age_groups_info: dict,
        mortality_of_disease_states_by_vulnerability_group: dict,
        diagnosis_of_disease_states_by_vulnerability_group: dict,
        hospitalization_of_disease_states_by_vulnerability_group: dict
    ):
        """
        """
        self.disease_states = disease_states
        self.vulnerability_groups = vulnerability_groups
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

        self.state_max_time = None

        self.state_time = 0

        self.live_state = 'alive'

        self.age = age

        # Determine age group
        self.determine_age_group()


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
        changed_state: bool=False
        ):
        """
        """
        if not changed_state:
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
        velocities_to_avoid: list
        ):
        """
        """
        #=============================================
        # Retrieve own velocity
        own_initial_velocity = np.array([self.vx, self.vy])
        norm_own_initial_velocity = np.sqrt(
            np.dot(own_initial_velocity, own_initial_velocity)
            )

        #=============================================
        # Create a vector to save new velocity vector
        new_velocity = np.array([0., 0.], dtype='float64')

        #=============================================
        # Avoid velocities
        for velocity_to_avoid in velocities_to_avoid:

            norm_velocity_to_avoid = np.sqrt(np.dot(velocity_to_avoid, velocity_to_avoid))

            # Find angle theta between both velocities
            costheta = np.dot(own_initial_velocity, velocity_to_avoid) / (norm_own_initial_velocity*norm_velocity_to_avoid)
            costheta = 1.0 if costheta > 1.0 else (-1.0 if costheta < -1.0 else costheta)

            v_parallel = own_initial_velocity * costheta
            v_perpendicular = - own_initial_velocity * np.sqrt(1.0 - costheta**2)

            #=============================================
            # Add up to new_velocity
            new_velocity += v_parallel + v_perpendicular

        #=============================================
        # Disaggregate new_velocity
        self.vx, self.vy = new_velocity


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
