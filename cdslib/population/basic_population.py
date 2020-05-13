import copy
import time
import numpy as np
import pandas as pd
from scipy import spatial
from cdslib.agents import AgentsInfo, Agent
from cdslib.agents.agent import determine_age_group

# TODO
# Improve diagnosis state?? (using networks)
#
# hospitalization, UCI
# Quarentine
# Parallellization
#
# Movility profiles? ... Clases of Agents
# Virus halo ... Probability density
# Add units
# 
# Change probability of transition state by exposition to virus

class BasicPopulation:
    """
    """
    def __init__(
        self,
        age_group_list: list,
        susceptibility_group_list: list,
        vulnerability_group_list: list,
        disease_state_list: list,
        population_age_groups_info: dict,
        initial_population_number: int,
        initial_population_age_distribution: dict,
        initial_population_distributions: dict
        ):
        """
        """
        self.susceptibility_groups: list = susceptibility_group_list

        self.vulnerability_groups: list = vulnerability_group_list

        self.disease_states: list = disease_state_list

        self.population_age_groups_info: dict = population_age_groups_info

        # Initialize population number
        self.step = 0
        self.initial_population_number: int = initial_population_number
        self.population_number: int = initial_population_number
        self.population_numbers: dict = {self.step: self.initial_population_number}

        default_columns = [
            'agent',
            'age',
            'age_group',
            'susceptibility_group',
            'vulnerability_group',
            'disease_state',
            'inmunization_level'
            ]

        self.initial_population_conditions = \
            pd.DataFrame(columns=default_columns)

        self.initial_population_conditions['agent'] = \
            np.arange(initial_population_number)

        age_values = [
            self.filling_age_function(
                initial_population_age_distribution.keys(),
                initial_population_age_distribution.values()
                )
            for x in range(initial_population_number)
            ]

        age_group_values = [
            determine_age_group(
                self.population_age_groups_info,
                age_values[x]
                )
            for x in range(initial_population_number)
            ]

        self.initial_population_conditions['age'] = age_values

        self.initial_population_conditions['age_group'] = age_group_values


        for nested_fields in initial_population_distributions.keys():

            dictionary = initial_population_distributions[nested_fields]

            # split fields
            fields = nested_fields.split('/')

            if len(fields) > 1:

                column_names = fields

                df = pd.DataFrame(columns=column_names)

                for (column_index, column_name) in enumerate(column_names):

                    if column_index != 0:

                        rows_number = df.shape[0]

                        column = eval(f'{column_name}_list')

                        aux_df = df.copy()

                        df = pd.DataFrame(columns=column_names)

                        for item in column:

                            array = [item for x in range(rows_number)]

                            aux_df[column_name] = array

                            df = df.append(aux_df, ignore_index=True)
                    else:
                        df[column_name] = eval(f'{column_name}_list')

                copy_df = df.copy()
                df['probability'] = [None for x in range(df.shape[0])]

                for row in copy_df.iterrows():
                    row_keys = list(row[1].keys())
                    row_items = row[1].to_list()

                    condition_list = [
                        f"(df['{key}'] == '{item}')"
                        for (key, item) in zip(row_keys, row_items)
                        ]

                    condition = ' & '.join(condition_list)

                    val = dictionary
                    for item in row_items:
                        val = val[item]

                    df.loc[eval(condition), 'probability'] = val

                for row in df.iterrows():
                    row_keys = list(row[1].keys())
                    row_items = row[1].to_list()

                    field = row_keys[-2]
                    filtering_fields = row_keys[:-2]
                    filtering_values = row_items[:-2]

                    condition_list = [
                        f"(df['{key}'] == '{item}')"
                        for (key, item) in zip(filtering_fields, filtering_values)
                        ]

                    condition = ' & '.join(condition_list)

                    values = df.loc[eval(condition), field].to_list()
                    probabilities = df.loc[eval(condition), 'probability'].to_list()

                    condition_list = [
                        f"(self.initial_population_conditions['{key}'] == '{item}')"
                        for (key, item) in zip(filtering_fields, filtering_values)
                        ]

                    condition = ' & '.join(condition_list)

                    agents_list = \
                        self.initial_population_conditions.loc[
                            eval(condition),
                            'agent'
                            ].to_list()

                    for agent_index in agents_list:
                        self.initial_population_conditions.loc[
                            self.initial_population_conditions['agent'] == agent_index,
                            field
                            ] = self.filling_function(values, probabilities)
            else:
                field = fields[0]

                values = [
                    self.filling_function(dictionary.keys(), dictionary.values())
                    for x in range(self.initial_population_number)
                    ]

                self.initial_population_conditions[field] = values


    def filling_age_function(
        self,
        values: list,
        probabilities: list
        ):
        """
        """
        # Throw the dice
        dice = np.random.random_sample()

        cummulative_probability = 0.

        for age_group, probability in zip(values, probabilities):

            cummulative_probability += probability

            if dice <= cummulative_probability:

                min_age = self.population_age_groups_info[age_group]['min_age']
                max_age = self.population_age_groups_info[age_group]['max_age']

                if max_age:
                    age_value = (max_age - min_age) * np.random.random_sample() + min_age
                else:
                    age_value = (80. - min_age) * np.random.exponential(scale=1./1.5) + min_age

                return age_value if age_value < 100. else 100.


    def filling_function(
        self,
        values,
        probabilities
        ):
        """
        """
        # Throw the dice
        dice = np.random.random_sample()
        
        cummulative_probability = 0.
        
        for value, probability in zip(values, probabilities):
            
            cummulative_probability += probability
            
            if dice <= cummulative_probability:

                return value


    def create_population(
        self,
        agents_info: AgentsInfo,
        horizontal_length: float,
        vertical_length: float,
        maximum_speed: float,
        maximum_free_random_speed: float,
        dt_scale_in_years: float
        ):
        """
        """
        # Initialize step
        self.step: int = 0
        self.dt: int = 1
        self.dt_scale_in_years: float = dt_scale_in_years

        # Initialize maximum free random speed
        self.maximum_free_random_speed: float = maximum_free_random_speed

        # Initialize horizontal and vertical length
        self.xmax: float = horizontal_length/2
        self.xmin: float = - horizontal_length/2
        self.ymax: float = vertical_length/2
        self.ymin: float = - vertical_length/2

        # Create list of agents
        self.population: dict = {}

        for agent_index in range(self.initial_population_number):

            agent_dict = self.initial_population_conditions.loc[
                self.initial_population_conditions['agent'] == agent_index
                ].to_dict(orient='records')[0]

            self.population[agent_index] = Agent(
                # Send AgentsInfo object
                agents_info=agents_info,

                # Define positions as random
                x=(self.xmax - self.xmin) * np.random.random_sample() \
                    + self.xmin,
                y=(self.ymax - self.ymin) * np.random.random_sample() \
                    + self.ymin,

                # Set up maximum speed
                vmax=maximum_speed,

                **agent_dict
                )

        # Initialize Pandas DataFrame
        keys = list(self.population[0].getkeys())
        keys.append('step')
        self.agents_info_df: pd.DataFrame = pd.DataFrame(columns = keys)

        # Populate DataFrame
        self.populate_df()


    def evolve_population(self):
        """
        """
        # Evolve step
        self.step += self.dt

        #=============================================
        # Change population states by means of state transition
        # and update diagnosis and hospitalization states

        # Cycle runs along all agents in current population
        for agent_index in self.population.keys():
            self.population[agent_index].disease_state_transition(self.dt)

        #=============================================
        # Check dead agents
        self.check_dead_agents()

        #=============================================
        # Update population numbers
        self.population_numbers[self.step] = self.population_number

        #=============================================
        # Create KDTree for agents of each state
        spatial_trees_by_disease_state, agents_indices_by_disease_state = \
            self.spatial_tress_by_disease_state()

        #=============================================
        # Create KDTree for agents of each state
        population_positions, population_velocities = \
            self.retrieve_population_positions_and_velocities()

        #=============================================
        # Update alertness states and avoid avoidable agents

        # Cycle runs along all agents in current population
        for agent_index in self.population.keys():

            self.population[agent_index].update_alertness_state(
                self.step,
                spatial_trees_by_disease_state,
                agents_indices_by_disease_state,
                population_positions,
                population_velocities
                )

        #=============================================
        # Change population states by means of contagion

        # Cycle runs along all agents in current population
        for agent_index in self.population.keys():

            self.population[agent_index].disease_state_transition_by_contagion(
                self.step,
                spatial_trees_by_disease_state,
                agents_indices_by_disease_state
                )

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
            dead = self.population[
                agent_index
                ].age_and_death_verification(self.dt_scale_in_years)

            if dead:
                population = self.remove_agent(
                    agent_index,
                    population
                    )

        # Copy population into self.population
        self.population = copy.deepcopy(population)


    def remove_agent(
        self,
        agent_index: int,
        population
        ):
        """
        """
        # Remove agent and catch its value using pop()
        removed_agent = population.pop(agent_index)

        # Add agent to agents_info_df
        agent_dict = removed_agent.getstate()

        self.populate_df(agent_dict)

        self.population_number -= 1

        return population


    def spatial_tress_by_disease_state(self):
        """
        """
        #=============================================
        # Create a dictionary for saving KDTree for agents of each state
        spatial_trees_by_disease_state = {}

        # Create a dictionary for saving indices for agents of each state
        agents_indices_by_disease_state = {}

        for disease_state in self.disease_states:

            # Spatial tree to calculate distances fastly for agents of each state
            points = [
                [self.population[agent_index].x, self.population[agent_index].y]
                for agent_index in self.population.keys()
                if self.population[agent_index].disease_state == disease_state
                ]

            agents_indices = np.array([
                agent_index
                for agent_index in self.population.keys()
                if self.population[agent_index].disease_state == disease_state
                ])

            spatial_trees_by_disease_state[disease_state] = \
                spatial.KDTree(points) if len(points) is not 0 else None

            agents_indices_by_disease_state[disease_state] = \
                agents_indices if len(agents_indices) is not 0 else None

        return spatial_trees_by_disease_state, agents_indices_by_disease_state


    def retrieve_population_positions_and_velocities(self):
        """
        """
        population_positions = {}
        population_velocities = {}

        # Cycle runs along all agents in current population
        for agent_index in self.population.keys():
            population_positions[agent_index] = np.array([
                self.population[agent_index].x,
                self.population[agent_index].y
                ])

            population_velocities[agent_index] = np.array([
                self.population[agent_index].vx,
                self.population[agent_index].vy
                ])

        return population_positions, population_velocities


    def populate_df(
        self,
        agent_dict: dict=None
        ):
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
