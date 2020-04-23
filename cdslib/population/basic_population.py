import copy
import time
import numpy as np
import pandas as pd
from scipy import spatial
from cdslib.agents import AgentsInfo, Agent

# TODO
# inmunization_level
# Quarentine
# Virus halo

class BasicPopulation:
    """
    """
    def __init__(
        self,
        disease_states: list,
        vulnerability_groups: list,
        contagion_dynamics_info: dict,
        social_distancing_info: dict
        ):

        self.disease_states = disease_states

        self.vulnerability_groups = vulnerability_groups

        # contagion_dynamics_info
        self.disease_states_transitions_by_contagion = \
            contagion_dynamics_info[
                'disease_states_transitions_by_contagion'
                ]

        self.contagion_probabilities_by_susceptibility_groups = \
            contagion_dynamics_info[
                'contagion_probabilities_by_susceptibility_groups'
                ]

        self.dynamics_of_disease_states_contagion = \
            contagion_dynamics_info[
                'dynamics_of_disease_states_contagion'
                ]

        self.inmunization_level_info = \
            contagion_dynamics_info[
                'inmunization_level_info'
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


    def create_population(
        self,
        agents_info: AgentsInfo,
        population_number: int,
        vulnerable_number: int, # TODO: This should be a list in order to manage a list of groups
        population_group_list: list,
        population_state_list: list, # TODO: Add a list with each agent's age
        horizontal_length: float,
        vertical_length: float,
        maximum_speed: float,
        maximum_free_random_speed: float,
        dt_scale_in_years: float
        ):
        """
        """
        # Initialize step
        self.step = 0
        self.dt = 1
        self.dt_scale_in_years = dt_scale_in_years

        # Initialize population number
        self.initial_population_number = population_number
        self.population_numbers = [self.initial_population_number]

        # Initialize vulnerable people number
        self.vulnerable_number = vulnerable_number

        # Initialize maximum free random speed
        self.maximum_free_random_speed = maximum_free_random_speed

        # Initialize horizontal and vertical length
        self.xmax = horizontal_length/2
        self.xmin = - horizontal_length/2
        self.ymax = vertical_length/2
        self.ymin = - vertical_length/2

        # Create list of agents
        self.population = {
            agent_index: Agent(
                # Send AgentsInfo object
                agents_info=agents_info,
                
                # Define label
                agent=agent_index,

                # Define positions as random
                x=(self.xmax - self.xmin) * np.random.random_sample() \
                    + self.xmin,
                y=(self.ymax - self.ymin) * np.random.random_sample() \
                    + self.ymin,

                # Set up maximum speed
                vmax=maximum_speed,

                # Define population group
                group=population_group_list[agent_index],

                # Define state
                state=population_state_list[agent_index],

                # Define agent's age
                age=20 # population_age_list[agent_index]

            ) for agent_index in range(self.initial_population_number)
        }

        # Initialize Pandas DataFrame
        keys = list(self.population[0].getkeys())
        keys.append('step')
        self.agents_info_df = pd.DataFrame(columns = keys)

        # Populate DataFrame
        self.populate_df()


    def evolve_population(self):
        """
        """
        # Evolve step
        self.step = self.step + self.dt

        #=============================================
        # Check dead agents
        self.check_dead_agents()

        #=============================================
        # Change population states by means of state transition
        # and update diagnosis and hospitalization states

        # Cycle runs along all agents in current population
        for agent_index in self.population.keys():
            self.population[agent_index].state_transition(self.dt)

        #=============================================
        # Update alertness states and avoid avoidable agents
        self.update_alertness_states()

        #=============================================
        # Change population states by means of contagion
        self.transition_by_contagion()

        #=============================================
        # Move agents

        # Cycle runs along all agents in current population
        for agent_index in self.population.keys():

            self.population[agent_index].move(
                self.dt,
                self.maximum_free_random_speed,
                self.xmin,
                self.xmax,
                self.ymin,
                self.ymax
                )

        #=============================================
        # Populate DataFrame
        self.populate_df()


    def check_dead_agents(self):
        """
        """     
        # Copy self.population using deepcopy in order to
        # prevent modifications on self.population
        population = copy.deepcopy(self.population)
        
        # Cycle runs along all agents in current population
        for agent_index in self.population.keys():
            
            # Check if agent is dead
            dead = population[
                agent_index
                ].age_and_death_verification(self.dt_scale_in_years)
            
            if dead:
                # Remove agent and catch its value using pop()
                removed_agent = population.pop(agent_index)
                
                # Add agent to agents_info_df
                agent_dict = removed_agent.getstate()
                
                self.populate_df(agent_dict)
                
        # Copy population into self.population
        self.population = copy.deepcopy(population)


    def spatial_tress_by_state(self):
        """
        """
        #=============================================
        # Create a dictionary for saving KDTree for agents of each state
        self.spatial_trees = {}

        # Create a dictionary for saving indices for agents of each state
        self.agents_indices_by_state = {}

        for state in self.disease_states_transitions_by_contagion.keys():
    
            # Spatial tree to calculate distances fastly for agents of each state
            points = [
                [self.population[agent_index].x, self.population[agent_index].y]
                for agent_index in self.population.keys()
                if self.population[agent_index].state == state
                ]

            agents_indices = np.array([
                agent_index
                for agent_index in self.population.keys()
                if self.population[agent_index].state == state
                ])

            self.spatial_trees[state] = \
                spatial.KDTree(points) if len(points) is not 0 else None

            self.agents_indices_by_state[state] = \
                agents_indices if len(agents_indices) is not 0 else None


    def transition_by_contagion(self):
        """
        """
        #=============================================
        # Create KDTree for agents of each state
        self.spatial_tress_by_state()

        #=============================================
        # Cycle runs along all agents in current population
        for agent_index in self.population.keys():

            # Agent can get infected ?
            if self.dynamics_of_disease_states_contagion[
                self.population[agent_index].state
                ]['can_get_infected']:

                # Retrieve agent location
                agent_location = [
                    self.population[agent_index].x,
                    self.population[agent_index].y
                    ]

                # List to save who infected the agent
                infected_by = []

                # Cycle through each spreader to see if the agent gets 
                # infected by the spreader
                for state in self.dynamics_of_disease_states_contagion.keys():

                    if (self.dynamics_of_disease_states_contagion[state]['can_spread']
                    and self.spatial_trees[state]):

                        # Detect if any spreader is inside a distance equal to
                        # the corresponding spread_radius
                        points_inside_radius = self.spatial_trees[state].query_ball_point(
                            agent_location,
                            self.dynamics_of_disease_states_contagion[state]['spread_radius']
                            )

                        spreaders_indices_inside_radius = \
                            self.agents_indices_by_state[state][points_inside_radius]
                        
                        # If agent_index in spreaders_indices_inside_radius,
                        # then remove it
                        if agent_index in spreaders_indices_inside_radius:

                            spreaders_indices_inside_radius = np.setdiff1d(
                                spreaders_indices_inside_radius,
                                agent_index
                                )

                        # Calculate joint probability for contagion
                        joint_probability = \
                            (1.0 - self.population[agent_index].inmunization_level) \
                            * self.contagion_probabilities_by_susceptibility_groups[
                                self.population[agent_index].group
                                ] \
                            * self.dynamics_of_disease_states_contagion[
                                state]['spread_probability']

                        # Check if got infected
                        for spreader_agent_index in spreaders_indices_inside_radius:
                            # Throw the dice
                            dice = np.random.random_sample()

                            if dice <= joint_probability:
                                # Got infected !!!
                                # Save who infected the agent
                                infected_by.append(spreader_agent_index)


                if len(infected_by) is not 0:
       
                    self.population[agent_index].infected_by = infected_by

                    # Verify: becomes into ? ... Throw the dice
                    dice = np.random.random_sample()

                    cummulative_probability = 0.

                    for (probability, becomes_into_state) in sorted(
                        zip(
                            self.disease_states_transitions_by_contagion[
                                self.population[agent_index].state]['transition_probability'],
                            self.disease_states_transitions_by_contagion[
                                self.population[agent_index].state]['becomes_into']
                            ),
                        # In order to be in descending order
                        reverse=True
                        ):
                        cummulative_probability += probability

                        if dice <= cummulative_probability:

                            self.population[agent_index].state = becomes_into_state

                            self.population[
                                agent_index
                                ].update_diagnosis_state(self.step, the_state_changed=True)

                            self.population[agent_index].state_time = 0

                            self.population[agent_index].determine_state_time()

                            break


    def update_alertness_states(self):
        """
        """
        #=============================================
        # Create KDTree for agents of each state
        self.spatial_tress_by_state()
        
        #=============================================
        # Cycle runs along all agents in current population
        for agent_index in self.population.keys():

            # Initialize agent alertness
            self.population[agent_index].alertness = False

            #=============================================
            # Agent should be alert?
            if self.dynamics_of_alertness_of_disease_states_by_vulnerability_group[
                self.population[agent_index].group
                ][self.population[agent_index].state]['should_be_alert']:
                
                # Retrieve agent location
                agent_location = [
                    self.population[agent_index].x,
                    self.population[agent_index].y
                    ]

                # Initialize "alerted by"
                self.population[agent_index].alerted_by = []

                # Cycle through each state of the neighbors to see if the agent
                # should be alert
                for state in self.disease_states_transitions_by_contagion.keys():

                    # Note that an 'avoidable_agent' depends on its state but
                    # also on each group, it means that each group defines
                    # which state is avoidable
                    if (self.dynamics_of_avoidance_of_disease_states_by_vulnerability_group[
                        self.population[agent_index].group][state]['avoidable_agent']
                        and self.spatial_trees[state]
                        ):
                        
                        # Detect if any avoidable agent is inside a distance
                        # equal to the corresponding spread_radius
                        points_inside_radius = self.spatial_trees[state].query_ball_point(
                            agent_location,
                            self.dynamics_of_avoidance_of_disease_states_by_vulnerability_group[
                                self.population[agent_index].group][state]['avoidness_radius']
                            )

                        avoidable_indices_inside_radius = \
                            self.agents_indices_by_state[state][points_inside_radius]
                        
                        # If agent_index in avoidable_indices_inside_radius,
                        # then remove it
                        if agent_index in avoidable_indices_inside_radius:
                            
                            avoidable_indices_inside_radius = np.setdiff1d(
                                avoidable_indices_inside_radius,
                                agent_index
                                )

                        if len(avoidable_indices_inside_radius) is not 0:
                            
                            for avoidable_agent_index in avoidable_indices_inside_radius:

                                # Must agent be alert ? ... Throw the dice
                                dice = np.random.random_sample()

                                # Note that alertness depends on a probability,
                                # which tries to model the probability that an
                                # agent with a defined group and state is alert
                                if (dice <= self.dynamics_of_alertness_of_disease_states_by_vulnerability_group[
                                    self.population[agent_index].group
                                    ][self.population[agent_index].state]['alertness_probability']
                                    ):

                                    # Agent is alerted !!!
                                    self.population[agent_index].alertness = True

                                    # Append avoidable_agent_index in alerted_by
                                    self.population[agent_index].alerted_by.append(
                                        avoidable_agent_index
                                        )

                #=============================================
                # Change movement direction if agent's alertness = True
                if self.population[agent_index].alertness:

                    # Retrieve positions of the agents to be avoided
                    positions_to_avoid = [
                        np.array(
                            [self.population[agent_to_be_avoided].x,
                            self.population[agent_to_be_avoided].y]
                            )
                        for agent_to_be_avoided in self.population[agent_index].alerted_by
                        ]

                    # Retrieve velocities of the agents to be avoided
                    velocities_to_avoid = [
                        np.array(
                            [self.population[agent_to_be_avoided].vx,
                            self.population[agent_to_be_avoided].vy]
                            )
                        for agent_to_be_avoided in self.population[agent_index].alerted_by
                        ]

                    # Change movement direction
                    self.population[agent_index].alert_avoid_agents(
                        positions_to_avoid,
                        velocities_to_avoid
                        )

    
    def populate_df(self, agent_dict: dict=None):
        """
        """
        if not agent_dict:
            # Cycle runs along all agents in current population
            population_array = []
            
            for agent_index in self.population.keys():
                agent_dict = self.population[agent_index].getstate()
                agent_dict['step'] = self.step
                
                population_array.append(agent_dict)
            
            self.agents_info_df = self.agents_info_df.append(
                population_array,
                ignore_index=True
                )
        else:
            # Append only received agent info
            agent_dict['step'] = self.step

            self.agents_info_df = self.agents_info_df.append(
                agent_dict,
                ignore_index=True
                )
