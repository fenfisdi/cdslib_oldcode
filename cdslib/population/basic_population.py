import os
import copy
import time
import functools
import numpy as np
import pandas as pd
from scipy import spatial

from multiprocessing import Pool, Manager

from cdslib.agents import AgentsInfo, Agent
from cdslib.agents.agent import determine_age_group

from cdslib.hospitals import Hospital

# TODO
#
# Parallelization
#
# Add units and time units
#
# Improve diagnosis state?? (using networks)
# Initial spatial distribution
# Family groups and networks
# Virus halo ... Probability density
# 
# Change probability of transition state by exposition to virus

class BasicPopulation:
    """
    """
    def __init__(
        self,
        hospitals_info: dict,

        quarantine_restrictions_info: dict,

        initial_population_number: int,

        population_age_groups_info: dict,
        age_groups: list,
        initial_population_age_distribution,
        population_age_initialization_mode: str,

        susceptibility_groups: list,
        vulnerability_groups: list,
        disease_states: list,
        initial_population_categorical_data_distributions,
        population_categorical_data_distribution_mode: str,

        initial_population_continous_data_constructor,
        population_continous_data_distribution_mode: str,

        initial_population_categorical_data_type2_constructor,
        population_categorical_data_type2_distribution_mode: str,

        n_processes: int=1
        ):
        """
        """
        self.n_processes = n_processes

        if self.n_processes > 1:
            self.pool = Pool(processes = n_processes)
            self.manager = Manager()

        #=============================================
        # Initialize Hospitals

        self.hospitals = {
            hospital_label: Hospital(**hospital_info)
            for (hospital_label, hospital_info) \
            in zip(hospitals_info.keys(), hospitals_info.values())
            }

        # Spatial tree to calculate distances fastly for hospitals
        hospitals_locations = [
            self.hospitals[hospital_label].get_hospital_location()
            for hospital_label in self.hospitals.keys()
            ]

        self.hospitals_spatial_tree = spatial.KDTree(hospitals_locations, leafsize=10)

        self.hospitals_labels = np.array([
            hospital_label
            for hospital_label in self.hospitals.keys()
            ])

        #=============================================
        # Quarantine restrictions info
        self.quarantine_restrictions_info = quarantine_restrictions_info

        self.restrictions_by_time_counter = None

        self.decreed_quarantine_by_hospitalization_restrictions = False

        self.decreed_quarantine_by_UCIs_restrictions = False

        self.decreed_quarantine_by_deaths = False
        self.time_after_decree_quarantine_by_deaths = None

        self.decreed_quarantine_by_diagnosed_people = False
        self.time_after_decree_quarantine_by_diagnosed_people = None

        #=============================================
        # Initialize population number

        self.step = 0
        self.initial_population_number: int = initial_population_number
        self.population_number: int = initial_population_number
        self.population_numbers: dict = {self.step: self.initial_population_number}


        #=============================================
        # Initialize void dataframe

        self.initial_population_conditions = \
            pd.DataFrame()

        self.initial_population_conditions['agent'] = \
            np.arange(initial_population_number)


        #=============================================
        # Initialize population age

        self.initialize_population_age_distribution(
            population_age_groups_info,
            age_groups,
            initial_population_age_distribution,
            population_age_initialization_mode
            )

        #=============================================
        # Initialize categorical fields

        self.susceptibility_groups: list = susceptibility_groups

        self.vulnerability_groups: list = vulnerability_groups

        self.disease_states: list = disease_states


        self.initialize_population_categorical_data_distribution(
            initial_population_categorical_data_distributions,
            population_categorical_data_distribution_mode
            )

        #=============================================
        # Initialize continous fields

        self.initialize_population_continous_data_distribution(
            initial_population_continous_data_constructor,
            population_continous_data_distribution_mode
            )

        #=============================================
        # Initialize categorical data type2 fields

        self.initialize_population_categorical_data_type2_distribution(
            initial_population_categorical_data_type2_constructor,
            population_categorical_data_type2_distribution_mode
            )


    def initialize_population_age_distribution(
        self,
        population_age_groups_info: dict,
        age_groups: list,
        initial_population_age_distribution,
        population_age_initialization_mode: str
        ):
        """
        """
        self.population_age_groups_info: dict = population_age_groups_info
        self.age_groups: dict = age_groups

        #=============================================
        # Mode validation

        valid_population_age_initialization_mode = {
            'categorical-json',
            'categorical-csv',
            'continuos age distribution-funct'
            }
        
        if population_age_initialization_mode not in valid_population_age_initialization_mode:
            ErrorString = (
                f'[ERROR] "population_age_initialization_mode" must be one of:'
                '\n'
                f'\t{valid_population_age_initialization_mode}'
                )
            raise ValueError(ErrorString)

        #=============================================
        # Initialize population age

        if population_age_initialization_mode == 'categorical-json':

            age_values = [
                self.filling_age_function(
                    initial_population_age_distribution.keys(),
                    initial_population_age_distribution.values()
                    )
                for i in range(self.initial_population_number)
                ]

            age_group_values = [
                determine_age_group(
                    self.population_age_groups_info,
                    age_values[i]
                    )
                for i in range(self.initial_population_number)
                ]

            self.initial_population_conditions['age'] = age_values

            self.initial_population_conditions['age_group'] = age_group_values


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


    def initialize_population_categorical_data_distribution(
        self,
        initial_population_categorical_data_distributions,
        population_categorical_data_distribution_mode: str
        ):
        """
        """
        #=============================================
        # Mode validation

        valid_population_categorical_data_distribution_mode = {'json', 'csv'}
        
        if population_categorical_data_distribution_mode not in valid_population_categorical_data_distribution_mode:
            ErrorString = (
                '[ERROR] "population_categorical_data_distribution_mode" must be one of:'
                '\n'
                f'\t{valid_population_categorical_data_distribution_mode}'
                )
            raise ValueError(ErrorString)

        #=============================================
        # Initialize population distributions

        if population_categorical_data_distribution_mode == 'json':

            for nested_fields in initial_population_categorical_data_distributions.keys():

                dictionary = initial_population_categorical_data_distributions[nested_fields]

                # split fields
                fields = nested_fields.split('/')

                if len(fields) > 1:

                    column_names = fields

                    df = pd.DataFrame(columns=column_names)

                    for (column_index, column_name) in enumerate(column_names):

                        if column_index != 0:

                            rows_number = df.shape[0]

                            column = eval(f'self.{column_name}s')

                            aux_df = df.copy()

                            df = pd.DataFrame(columns=column_names)

                            for item in column:

                                array = [item for x in range(rows_number)]

                                aux_df[column_name] = array

                                df = df.append(aux_df, ignore_index=True)
                        else:
                            df[column_name] = eval(f'self.{column_name}s')

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
                            f"(df['{key}'] == '{value}')"
                            for (key, value) in zip(filtering_fields, filtering_values)
                            ]

                        condition = ' & '.join(condition_list)

                        values = df.loc[eval(condition), field].to_list()
                        probabilities = df.loc[eval(condition), 'probability'].to_list()

                        condition_list = [
                            f"(self.initial_population_conditions['{key}'] == '{value}')"
                            for (key, value) in zip(filtering_fields, filtering_values)
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


    def initialize_population_continous_data_distribution(
        self,
        initial_population_continous_data_constructor,
        population_continous_data_distribution_mode: str
        ):
        """
        """
        #=============================================
        # Mode validation

        valid_population_continous_data_distribution_mode = {'json', 'csv'}
        
        if population_continous_data_distribution_mode not in valid_population_continous_data_distribution_mode:
            ErrorString = (
                f'[ERROR] "population_continous_data_distribution_mode" must be one of:'
                '\n'
                f'\t{valid_population_continous_data_distribution_mode}'
                )
            raise ValueError(ErrorString)

        #=============================================
        # Initialize population distributions

        if population_continous_data_distribution_mode == 'json':

            for field in initial_population_continous_data_constructor.keys():

                nested_categorical_fields = \
                    initial_population_continous_data_constructor[field]['nested_categorical_fields']

                nested_continous_fields = \
                    initial_population_continous_data_constructor[field]['nested_continous_fields']

                probability_distribution_function = \
                    initial_population_continous_data_constructor[field]['probability_distribution_function']

                if len(nested_categorical_fields) is not 0:

                    df = pd.DataFrame()

                    for (column_index, column_name) in enumerate(nested_categorical_fields):

                        if column_index != 0:

                            rows_number = df.shape[0]

                            column = eval(f'self.{column_name}s')

                            aux_df = df.copy()

                            df = pd.DataFrame()

                            for item in column:

                                array = [item for x in range(rows_number)]

                                aux_df[column_name] = array

                                df = df.append(aux_df, ignore_index=True)
                        else:
                            df[column_name] = eval(f'self.{column_name}s')


                    categorical_filtering_fields = nested_categorical_fields

                    for row in df.iterrows():

                        categorical_filtering_values = row[1].to_list()

                        condition_list = [
                            f"(self.initial_population_conditions['{key}'] == '{value}')"
                            for (key, value) in zip(
                                categorical_filtering_fields,
                                categorical_filtering_values
                                )
                            ]

                        condition = ' & '.join(condition_list)

                        agents_list = \
                            self.initial_population_conditions.loc[
                                eval(condition),
                                'agent'
                                ].to_list()

                        arg_list = [
                            f"{key} = '{value}'"
                            for (key, value) in zip(
                                categorical_filtering_fields,
                                categorical_filtering_values
                                )
                            ]
                        categorical_arg_string = ', '.join(arg_list)

                        for agent_index in agents_list:

                            if not nested_continous_fields:

                                self.initial_population_conditions.loc[
                                    self.initial_population_conditions['agent'] == agent_index,
                                    field
                                    ] = eval(
                                        f"probability_distribution_function({categorical_arg_string})"
                                        )
                            else:
                                df = self.initial_population_conditions.loc[
                                    self.initial_population_conditions['agent'] == agent_index,
                                    field
                                    ].copy()

                                agent_dict = df.to_dict(orient='records')

                                arg_list = [
                                    f"{key} = '{value}'"
                                    for (key, value) in zip(agent_dict.keys(), agent_dict.values())
                                    if key in nested_continous_fields
                                    ]
                                continous_arg_string = ', '.join(arg_list)

                                self.initial_population_conditions.loc[
                                    self.initial_population_conditions['agent'] == agent_index,
                                    field
                                    ] = eval(
                                        "probability_distribution_function("
                                        f"{categorical_arg_string}, {continous_arg_string})"
                                        )
                else:
                    if not nested_continous_fields:

                        values = [
                            probability_distribution_function()
                            for x in range(self.initial_population_number)
                            ]

                        self.initial_population_conditions[field] = values

                    else:
                        agents_list = \
                            self.initial_population_conditions['agent'].to_list()

                        values = []

                        for agent_index in agents_list:

                            df = self.initial_population_conditions.loc[
                                self.initial_population_conditions['agent'] == agent_index,
                                field
                                ].copy()

                            agent_dict = df.to_dict(orient='records')

                            arg_list = [
                                f"{key} = '{value}'"
                                for (key, value) in zip(agent_dict.keys(), agent_dict.values())
                                if key in nested_continous_fields
                                ]
                            continous_arg_string = ', '.join(arg_list)

                            value = eval(
                                    f"probability_distribution_function({continous_arg_string})"
                                    )
                            values.append(value)

                        self.initial_population_conditions[field] = values


    def initialize_population_categorical_data_type2_distribution(
        self,
        initial_population_categorical_data_type2_constructor,
        population_categorical_data_type2_distribution_mode: str
        ):
        """
        """
        #=============================================
        # Mode validation

        valid_population_categorical_data_type2_distribution_mode = {'json', 'csv'}
        
        if population_categorical_data_type2_distribution_mode not in valid_population_categorical_data_type2_distribution_mode:
            ErrorString = (
                f'[ERROR] "population_categorical_data_type2_distribution_mode" must be one of:'
                '\n'
                f'\t{valid_population_categorical_data_type2_distribution_mode}'
                )
            raise ValueError(ErrorString)

        #=============================================
        # Initialize population distributions

        if population_categorical_data_type2_distribution_mode == 'json':

            for field in initial_population_categorical_data_type2_constructor.keys():

                nested_categorical_fields = \
                    initial_population_categorical_data_type2_constructor[field]['nested_categorical_fields']

                nested_continous_fields = \
                    initial_population_categorical_data_type2_constructor[field]['nested_continous_fields']

                probability_distribution_function = \
                    initial_population_categorical_data_type2_constructor[field]['probability_distribution_function']

                if len(nested_categorical_fields) is not 0:

                    df = pd.DataFrame()

                    for (column_index, column_name) in enumerate(nested_categorical_fields):

                        if column_index != 0:

                            rows_number = df.shape[0]

                            column = eval(f'self.{column_name}s')

                            aux_df = df.copy()

                            df = pd.DataFrame()

                            for item in column:

                                array = [item for x in range(rows_number)]

                                aux_df[column_name] = array

                                df = df.append(aux_df, ignore_index=True)
                        else:
                            df[column_name] = eval(f'self.{column_name}s')


                    categorical_filtering_fields = nested_categorical_fields

                    for row in df.iterrows():

                        categorical_filtering_values = row[1].to_list()

                        condition_list = [
                            f"(self.initial_population_conditions['{key}'] == '{value}')"
                            for (key, value) in zip(
                                categorical_filtering_fields,
                                categorical_filtering_values
                                )
                            ]

                        condition = ' & '.join(condition_list)

                        agents_list = \
                            self.initial_population_conditions.loc[
                                eval(condition),
                                'agent'
                                ].to_list()

                        arg_list = [
                            f"{key} = '{value}'"
                            for (key, value) in zip(
                                categorical_filtering_fields,
                                categorical_filtering_values
                                )
                            ]
                        categorical_arg_string = ', '.join(arg_list)

                        for agent_index in agents_list:

                            if not nested_continous_fields:

                                self.initial_population_conditions.loc[
                                    self.initial_population_conditions['agent'] == agent_index,
                                    field
                                    ] = eval(
                                        f"probability_distribution_function({categorical_arg_string})"
                                        )
                            else:
                                df = self.initial_population_conditions.loc[
                                    self.initial_population_conditions['agent'] == agent_index
                                    ].copy()

                                agent_dict_list = df.to_dict(orient='records')

                                agent_dict = agent_dict_list[0]

                                arg_list = [
                                    f"{key} = float('{value}')"
                                    for (key, value) in zip(agent_dict.keys(), agent_dict.values())
                                    if key in nested_continous_fields
                                    ]
                                continous_arg_string = ', '.join(arg_list)

                                self.initial_population_conditions.loc[
                                    self.initial_population_conditions['agent'] == agent_index,
                                    field
                                    ] = eval(
                                        "probability_distribution_function("
                                        f"{categorical_arg_string}, {continous_arg_string})"
                                        )
                else:
                    if not nested_continous_fields:
                        values = [
                            probability_distribution_function()
                            for x in range(self.initial_population_number)
                            ]

                        self.initial_population_conditions[field] = values
                    else:
                        agents_list = \
                            self.initial_population_conditions['agent'].to_list()

                        values = []

                        for agent_index in agents_list:

                            df = self.initial_population_conditions.loc[
                                self.initial_population_conditions['agent'] == agent_index,
                                field
                                ].copy()

                            agent_dict = df.to_dict(orient='records')

                            arg_list = [
                                f"{key} = '{value}'"
                                for (key, value) in zip(agent_dict.keys(), agent_dict.values())
                                if key in nested_continous_fields
                                ]
                            continous_arg_string = ', '.join(arg_list)

                            value = eval(
                                    f"probability_distribution_function({continous_arg_string})"
                                    )
                            values.append(value)

                        self.initial_population_conditions[field] = values


    def create_population(
        self,
        agents_info: AgentsInfo,
        horizontal_length: float,
        vertical_length: float,
        maximum_free_random_speed_factor: float,
        dt_scale_in_years: float,
        dt_in_time_units: str,
        init_datetime: str
        ):
        """
        """
        # Initialize step
        self.step: int = 0
        self.datetime = pd.to_datetime(init_datetime)
        self.dt: int = 1
        self.dt_scale_in_years: float = dt_scale_in_years
        self.dt_timedelta = pd.to_timedelta(dt_in_time_units)
        

        # Initialize maximum free random speed
        self.maximum_free_random_speed_factor: float = maximum_free_random_speed_factor

        # Initialize horizontal and vertical length
        self.xmax: float = horizontal_length/2
        self.xmin: float = - horizontal_length/2
        self.ymax: float = vertical_length/2
        self.ymin: float = - vertical_length/2

        # Create list of agents
        if self.n_processes == 1:
            self.population: dict = {}
        else:
            self.population = self.manager.dict()


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

                **agent_dict
                )

            self.hospitals = \
                self.population[agent_index].update_hospitalization_state(
                    self.dt,
                    self.hospitals,
                    self.hospitals_spatial_tree,
                    self.hospitals_labels
                    )

        # Restrictions by time dataframe
        if self.quarantine_restrictions_info[
            'restrictions_by_time']['enabled']:
            cols = ['step', 'datetime']
            cols.extend(
                self.quarantine_restrictions_info[
                'restrictions_by_time']['quarantine_times_by_group'].keys()
                )
            self.groups_in_quarantine_by_time_restrictions_df = \
                pd.DataFrame(columns = cols)

        # Restrictions by hospitals variables dataframe
        if self.quarantine_restrictions_info[
            'restictions_by_hospitals_variables']['enabled']:

            # Variable 1: Hospitals capacity
            if self.quarantine_restrictions_info[
                'restictions_by_hospitals_variables'][
                'conditions_for_quarantine']['hospitals_capacity']['enabled']:
                cols = ['step', 'datetime']
                cols.extend(
                    self.quarantine_restrictions_info[
                    'restictions_by_hospitals_variables'][
                    'conditions_for_quarantine']['hospitals_capacity']['groups_target_to_quarantine']
                    )
                self.groups_in_quarantine_by_hospitalization_restrictions_df = \
                    pd.DataFrame(columns = cols)

            # Variable 2: UCIs capacity
            if self.quarantine_restrictions_info[
                'restictions_by_hospitals_variables'][
                'conditions_for_quarantine']['UCIs_capacity']['enabled']:
                cols = ['step', 'datetime']
                cols.extend(
                    self.quarantine_restrictions_info[
                    'restictions_by_hospitals_variables'][
                    'conditions_for_quarantine']['UCIs_capacity']['groups_target_to_quarantine']
                    )
                self.groups_in_quarantine_by_UCIs_restrictions_df = \
                    pd.DataFrame(columns = cols)

        # Restrictions by population variables dataframe
        if self.quarantine_restrictions_info[
            'restictions_by_population_variables']['enabled']:

            # Variable 1: 'dead by disease'
            if self.quarantine_restrictions_info[
                'restictions_by_population_variables'][
                'conditions_for_quarantine']['dead by disease']['enabled']:
                cols = ['step', 'datetime']
                cols.extend(
                    self.quarantine_restrictions_info[
                    'restictions_by_population_variables'][
                    'conditions_for_quarantine']['dead by disease']['groups_target_to_quarantine']
                    )
                self.groups_in_quarantine_by_deaths_df = \
                    pd.DataFrame(columns = cols)

            # Variable 2: 'diagnosed'
            if self.quarantine_restrictions_info[
                'restictions_by_population_variables'][
                'conditions_for_quarantine']['diagnosed']['enabled']:
                cols = ['step', 'datetime']
                cols.extend(
                    self.quarantine_restrictions_info[
                    'restictions_by_population_variables'][
                    'conditions_for_quarantine']['diagnosed']['groups_target_to_quarantine']
                    )
                self.groups_in_quarantine_by_diagnosed_people_df = \
                    pd.DataFrame(columns = cols)

        # Quarantine
        self.population_quarantine()

        # Initialize Pandas DataFrame
        keys = ['step', 'datetime']
        keys.extend(list(self.population[0].getkeys()))
        self.agents_info_df: pd.DataFrame = pd.DataFrame(columns = keys)

        # Populate DataFrame
        self.populate_agents_info_df()


    def evolve_population(
        self,
        nsteps: int,
        step_monitoring: bool=False,
        time_monitoring: bool=False,
        export_population_df: bool=False,
        population_df_filename: str='agents_info_df',
        population_df_path: str='.'
        ):
        """
        """
        if time_monitoring:
            start_time = time.time()

        if step_monitoring:
            #=========================
            # Evolve population cycle
            for i in range(nsteps):
                print(f'step: {i}')
                self.evolve_population_single_step()
        else:
            #=========================
            # Evolve population cycle
            for i in range(nsteps):
                self.evolve_population_single_step()

        if time_monitoring:
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            elapsed_time_remainder_str = '{:.10f}'.format(elapsed_time % 1000)
            elapsed_time_remainder = elapsed_time_remainder_str.split('.')[1]

            print(
                '\n'
                'Time elapsed: '
                f'{time.strftime(f"%H:%M:%S.{elapsed_time_remainder}", time.gmtime(elapsed_time))}'
                )

        #=========================
        # Export population dataframe
        if export_population_df:
            
            filename = os.path.join(population_df_path, population_df_filename)
            
            if time_monitoring:
                start_time = time.time()
                
                self.agents_info_df.to_csv(filename + '.csv', index=False)

                #===============================================================
                # Export quarantine dataframes

                # Restrictions by time dataframe
                if self.quarantine_restrictions_info[
                    'restrictions_by_time']['enabled']:
                    self.groups_in_quarantine_by_time_restrictions_df.to_csv(
                        filename + '_quarantine_by_time_restrictions_df' + '.csv',
                        index=False
                        )

                # Restrictions by hospitals variables dataframe
                if self.quarantine_restrictions_info[
                    'restictions_by_hospitals_variables']['enabled']:

                    # Variable 1: Hospitals capacity
                    if self.quarantine_restrictions_info[
                        'restictions_by_hospitals_variables'][
                        'conditions_for_quarantine']['hospitals_capacity']['enabled']:
                        self.groups_in_quarantine_by_hospitalization_restrictions_df.to_csv(
                            filename + '_quarantine_by_hospitalization' + '.csv',
                            index=False
                            )

                    # Variable 2: UCIs capacity
                    if self.quarantine_restrictions_info[
                        'restictions_by_hospitals_variables'][
                        'conditions_for_quarantine']['UCIs_capacity']['enabled']:
                        self.groups_in_quarantine_by_UCIs_restrictions_df.to_csv(
                            filename + '_quarantine_by_UCIs' + '.csv',
                            index=False
                            )

                # Restrictions by population variables dataframe
                if self.quarantine_restrictions_info[
                    'restictions_by_population_variables']['enabled']:

                    # Variable 1: 'dead by disease'
                    if self.quarantine_restrictions_info[
                        'restictions_by_population_variables'][
                        'conditions_for_quarantine']['dead by disease']['enabled']:
                        self.groups_in_quarantine_by_deaths_df.to_csv(
                            filename + '_quarantine_by_deaths' + '.csv',
                            index=False
                            )

                    # Variable 2: 'diagnosed'
                    if self.quarantine_restrictions_info[
                        'restictions_by_population_variables'][
                        'conditions_for_quarantine']['diagnosed']['enabled']:
                        self.groups_in_quarantine_by_diagnosed_people_df.to_csv(
                            filename + '_quarantine_by_diagnosed_people' + '.csv',
                            index=False
                            )

                #===============================================================

                end_time = time.time()
                elapsed_time = end_time - start_time
                
                elapsed_time_remainder_str = '{:.10f}'.format(elapsed_time % 1000)
                elapsed_time_remainder = elapsed_time_remainder_str.split('.')[1]

                print(
                    '\n'
                    'Time elapsed: '
                    f'{time.strftime(f"%H:%M:%S.{elapsed_time_remainder}", time.gmtime(elapsed_time))}'
                    )
            else:
                self.agents_info_df.to_csv(filename + '.csv', index=False)

                #===============================================================
                # Export quarantine dataframes

                # Restrictions by time dataframe
                if self.quarantine_restrictions_info[
                    'restrictions_by_time']['enabled']:
                    self.groups_in_quarantine_by_time_restrictions_df.to_csv(
                        filename + '_quarantine_by_time_restrictions_df' + '.csv',
                        index=False
                        )

                # Restrictions by hospitals variables dataframe
                if self.quarantine_restrictions_info[
                    'restictions_by_hospitals_variables']['enabled']:

                    # Variable 1: Hospitals capacity
                    if self.quarantine_restrictions_info[
                        'restictions_by_hospitals_variables'][
                        'conditions_for_quarantine']['hospitals_capacity']['enabled']:
                        self.groups_in_quarantine_by_hospitalization_restrictions_df.to_csv(
                            filename + '_quarantine_by_hospitalization' + '.csv',
                            index=False
                            )

                    # Variable 2: UCIs capacity
                    if self.quarantine_restrictions_info[
                        'restictions_by_hospitals_variables'][
                        'conditions_for_quarantine']['UCIs_capacity']['enabled']:
                        self.groups_in_quarantine_by_UCIs_restrictions_df.to_csv(
                            filename + '_quarantine_by_UCIs' + '.csv',
                            index=False
                            )

                # Restrictions by population variables dataframe
                if self.quarantine_restrictions_info[
                    'restictions_by_population_variables']['enabled']:

                    # Variable 1: 'dead by disease'
                    if self.quarantine_restrictions_info[
                        'restictions_by_population_variables'][
                        'conditions_for_quarantine']['dead by disease']['enabled']:
                        self.groups_in_quarantine_by_deaths_df.to_csv(
                            filename + '_quarantine_by_deaths' + '.csv',
                            index=False
                            )

                    # Variable 2: 'diagnosed'
                    if self.quarantine_restrictions_info[
                        'restictions_by_population_variables'][
                        'conditions_for_quarantine']['diagnosed']['enabled']:
                        self.groups_in_quarantine_by_diagnosed_people_df.to_csv(
                            filename + '_quarantine_by_diagnosed_people' + '.csv',
                            index=False
                            )

                #===============================================================


    def evolve_population_single_step(self):
        """
        """
        # Evolve step
        self.step += self.dt

        # Evolve date
        self.datetime += self.dt_timedelta

        #=============================================
        # Change population states by means of state transition
        # and update diagnosis and hospitalization states

        # Cycle runs along all agents in current population
        if self.n_processes == 1:
            for agent_index in self.population.keys():
                self.hospitals = \
                    self.population[agent_index].disease_state_transition(
                        self.dt,
                        self.hospitals,
                        self.hospitals_spatial_tree,
                        self.hospitals_labels
                        )
        else:
            self.pool.starmap(
                    parallel_disease_state_transition,
                    [agent_index for agent_index in self.population.keys()]
                )

        #=============================================
        # Check dead agents
        self.check_dead_agents()

        #=============================================
        # Update population numbers
        self.population_numbers[self.step] = self.population_number

        #=============================================
        # Quarantine
        self.population_quarantine()

        #=============================================
        # Create KDTree for agents of each state
        spatial_trees_by_disease_state, agents_indices_by_disease_state = \
            self.spatial_tress_by_disease_state()

        #=============================================
        # Create KDTree for agents of each state
        population_positions, population_velocities = \
            self.retrieve_population_positions_and_velocities()

        #=============================================
        # Trace_agents' neighbors

        # Cycle runs along all agents in current population
        for agent_index in self.population.keys():

            self.population[agent_index].trace_neighbors(
                spatial_trees_by_disease_state,
                agents_indices_by_disease_state
                )

        #=============================================
        # Update alertness states and avoid avoidable agents

        # Cycle runs along all agents in current population
        for agent_index in self.population.keys():

            self.population[agent_index].update_alertness_state(
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
                self.dt,
                spatial_trees_by_disease_state,
                agents_indices_by_disease_state
                )

        #=============================================
        # Move agents

        # Cycle runs along all agents in current population
        for agent_index in self.population.keys():

            self.population[agent_index].move(
                self.dt,
                self.maximum_free_random_speed_factor,
                self.xmin,
                self.xmax,
                self.ymin,
                self.ymax
                )

        #=============================================
        # Populate Agents' DataFrame
        self.populate_agents_info_df()


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
            else:
                population[agent_index] = copy.deepcopy(self.population[agent_index])


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

        self.populate_agents_info_df(agent_dict)

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


        if self.n_processes == 1:

            for disease_state in self.disease_states:

                spatial_trees_and_agents_indices(
                    self.population,
                    self.population_number,
                    disease_state,
                    spatial_trees_by_disease_state,
                    agents_indices_by_disease_state
                    )

        elif self.n_processes > len(self.disease_states):

            pool = Pool(processes=len(self.disease_states))

            parallel_spatial_trees_by_disease_state = self.manager.dict()
            parallel_agents_indices_by_disease_state = self.manager.dict()

            pool.starmap(
                spatial_trees_and_agents_indices,
                [
                    (
                        self.population,
                        self.population_number,
                        disease_state,
                        parallel_spatial_trees_by_disease_state,
                        parallel_agents_indices_by_disease_state
                        )
                    for disease_state in self.disease_states
                    ]
                )

            spatial_trees_by_disease_state = \
                parallel_spatial_trees_by_disease_state._getvalue()
            agents_indices_by_disease_state = \
                parallel_agents_indices_by_disease_state._getvalue()

        else:
            pool = Pool(processes=self.n_processes)

            manager = Manager()
            parallel_spatial_trees_by_disease_state = self.manager.dict()
            parallel_agents_indices_by_disease_state = self.manager.dict()

            pool.starmap(
                spatial_trees_and_agents_indices,
                [
                    (
                        self.population,
                        self.population_number,
                        disease_state,
                        parallel_spatial_trees_by_disease_state,
                        parallel_agents_indices_by_disease_state
                        )
                    for disease_state in self.disease_states
                    ]
                )

            spatial_trees_by_disease_state = \
                parallel_spatial_trees_by_disease_state._getvalue()
            agents_indices_by_disease_state = \
                parallel_agents_indices_by_disease_state._getvalue()

        return spatial_trees_by_disease_state, agents_indices_by_disease_state


    def retrieve_population_positions_and_velocities(self):
        """
        """
        population_positions = {}
        population_velocities = {}

        # Cycle runs along all agents in current population
        for agent_index in self.population.keys():
            if not self.population[agent_index].is_hospitalized:

                population_positions[agent_index] = np.array([
                    self.population[agent_index].x,
                    self.population[agent_index].y
                    ])

                population_velocities[agent_index] = np.array([
                    self.population[agent_index].vx,
                    self.population[agent_index].vy
                    ])

        return population_positions, population_velocities


    def populate_agents_info_df(
        self,
        agent_dict: dict=None
        ):
        """
        """
        if not agent_dict:

            if self.n_processes == 1:
                # Cycle runs along all agents in current population
                population_array = []
                
                for agent_index in self.population.keys():
                    population_array.append(
                        retrieve_agent_dict(
                            self.population,
                            agent_index,
                            self.step,
                            self.datetime
                            )
                        )
            else:
                population_array = self.pool.starmap(
                    retrieve_agent_dict,
                    [
                        (self.population, agent_index, self.step, self.datetime)
                        for agent_index in self.population.keys()
                    ]
                )

            self.agents_info_df = self.agents_info_df.append(
                population_array,
                ignore_index=True
                )
        else:
            # Append only received agent info
            agent_dict['step'] = self.step
            agent_dict['datetime'] = self.datetime

            self.agents_info_df = self.agents_info_df.append(
                agent_dict,
                ignore_index=True
                )


    def population_quarantine(self):
        """
        """
        self.decreed_quarantine = False
        self.groups_in_quarantine = []

        #=============================================
        # Restrictions by time
        if self.quarantine_restrictions_info[
            'restrictions_by_time']['enabled']:

            groups_in_quarantine_by_time_restrictions = []

            if self.step == \
            self.quarantine_restrictions_info[
                'restrictions_by_time']['quarantine_start_time']:
                self.restrictions_by_time_counter = 0
            
            if self.step > \
            self.quarantine_restrictions_info[
                'restrictions_by_time']['quarantine_start_time']:
                self.restrictions_by_time_counter += 1

            if self.restrictions_by_time_counter is not None:
                for group in self.quarantine_restrictions_info[
                    'restrictions_by_time']['quarantine_times_by_group'].keys():

                    quarantine_start =  self.quarantine_restrictions_info[
                        'restrictions_by_time']['quarantine_times_by_group'][group][
                        'delay_on_quarantine_start']

                    quarantine_length = self.quarantine_restrictions_info[
                        'restrictions_by_time']['quarantine_times_by_group'][
                        group]['quarantine_length']
                    
                    time_without_quarantine = self.quarantine_restrictions_info[
                        'restrictions_by_time']['quarantine_times_by_group'][
                        group]['time_without_quarantine']

                    period = quarantine_length + time_without_quarantine
                    
                    group_time = self.restrictions_by_time_counter - quarantine_start

                    if 0 <= group_time:
                        group_time_mod = group_time % (period)

                        if group_time_mod < quarantine_length:

                            # Quarantine must be decreed

                            self.decreed_quarantine = True

                            groups_in_quarantine_by_time_restrictions.append(group)

            #=============================================
            # Extend self.groups_in_quarantine

            self.groups_in_quarantine.extend(groups_in_quarantine_by_time_restrictions)

            self.groups_in_quarantine = list(set(self.groups_in_quarantine))

            #=============================================
            # Populate groups_in_quarantine_by_time_restrictions_df

            # using fromkeys() method 
            quarantine_data_dict = dict.fromkeys(
                self.groups_in_quarantine_by_time_restrictions_df.columns.to_list(),
                None
                )
            quarantine_data_dict['step'] = self.step
            quarantine_data_dict['datetime'] = self.datetime

            for group in self.quarantine_restrictions_info[
            'restrictions_by_time']['quarantine_times_by_group'].keys():
                quarantine_data_dict[group] = 1 \
                    if group in groups_in_quarantine_by_time_restrictions \
                    else 0

            self.groups_in_quarantine_by_time_restrictions_df = \
                self.groups_in_quarantine_by_time_restrictions_df.append(
                    quarantine_data_dict, ignore_index=True
                    )

        #=============================================
        # Restrictions by hospitals variables
        if self.quarantine_restrictions_info[
            'restictions_by_hospitals_variables']['enabled']:

            #=============================================
            # Variable 1: Hospitals capacity
            if self.quarantine_restrictions_info[
                'restictions_by_hospitals_variables'][
                'conditions_for_quarantine']['hospitals_capacity']['enabled']:

                groups_in_quarantine_by_hospitalization_restrictions = []

                percentage_level_to_start_quarantine = self.quarantine_restrictions_info[
                    'restictions_by_hospitals_variables'][
                    'conditions_for_quarantine']['hospitals_capacity']['percentage_level_to_start_quarantine']

                percentage_level_to_exit_quarantine = self.quarantine_restrictions_info[
                    'restictions_by_hospitals_variables'][
                    'conditions_for_quarantine']['hospitals_capacity']['percentage_level_to_exit_quarantine']
                
                groups_target_to_quarantine = self.quarantine_restrictions_info[
                    'restictions_by_hospitals_variables'][
                    'conditions_for_quarantine']['hospitals_capacity']['groups_target_to_quarantine']

                max_hospitals_capacity = functools.reduce(
                    lambda x, y: x+y,
                    [hospital.maximum_hospitalization_capacity
                    for hospital in self.hospitals.values()]
                    )

                current_hospitals_capacity = functools.reduce(
                    lambda x, y: x+y,
                    [hospital.current_hospitalization_capacity
                    for hospital in self.hospitals.values()]
                    )

                percentage_level_capacity = current_hospitals_capacity/max_hospitals_capacity

                if not self.decreed_quarantine_by_hospitalization_restrictions:

                    if percentage_level_capacity <= percentage_level_to_start_quarantine:

                        # Quarantine must be decreed

                        self.decreed_quarantine = True

                        groups_in_quarantine_by_hospitalization_restrictions = \
                            groups_target_to_quarantine

                        self.decreed_quarantine_by_hospitalization_restrictions = True

                else:
                    # self.decreed_quarantine_by_hospitalization_restrictions == True

                    if not (percentage_level_capacity > percentage_level_to_exit_quarantine):

                            # Quarantine stills being decreed

                            self.decreed_quarantine = True

                            groups_in_quarantine_by_hospitalization_restrictions = \
                                groups_target_to_quarantine

                    else:
                        self.decreed_quarantine_by_hospitalization_restrictions = False

                #=============================================
                # Extend self.groups_in_quarantine

                self.groups_in_quarantine.extend(groups_in_quarantine_by_hospitalization_restrictions)

                self.groups_in_quarantine = list(set(self.groups_in_quarantine))

                #=============================================
                # Populate groups_in_quarantine_by_hospitalization_restrictions_df

                # using fromkeys() method
                if self.decreed_quarantine_by_hospitalization_restrictions:
                    quarantine_data_dict = dict.fromkeys(
                        self.groups_in_quarantine_by_hospitalization_restrictions_df.columns.to_list(),
                        1
                        )
                else:
                    quarantine_data_dict = dict.fromkeys(
                        self.groups_in_quarantine_by_hospitalization_restrictions_df.columns.to_list(),
                        0
                        )
                quarantine_data_dict['step'] = self.step
                quarantine_data_dict['datetime'] = self.datetime

                self.groups_in_quarantine_by_hospitalization_restrictions_df = \
                    self.groups_in_quarantine_by_hospitalization_restrictions_df.append(
                        quarantine_data_dict, ignore_index=True
                        )

            #=============================================
            # Variable 2: UCIs capacity
            if self.quarantine_restrictions_info[
                'restictions_by_hospitals_variables'][
                'conditions_for_quarantine']['UCIs_capacity']['enabled']:

                groups_in_quarantine_by_UCIs_restrictions = []

                percentage_level_to_start_quarantine = self.quarantine_restrictions_info[
                    'restictions_by_hospitals_variables'][
                    'conditions_for_quarantine']['UCIs_capacity']['percentage_level_to_start_quarantine']

                percentage_level_to_exit_quarantine = self.quarantine_restrictions_info[
                    'restictions_by_hospitals_variables'][
                    'conditions_for_quarantine']['UCIs_capacity']['percentage_level_to_exit_quarantine']
                
                groups_target_to_quarantine = self.quarantine_restrictions_info[
                    'restictions_by_hospitals_variables'][
                    'conditions_for_quarantine']['UCIs_capacity']['groups_target_to_quarantine']

                max_UCIs_capacity = functools.reduce(
                    lambda x, y: x+y,
                    [hospital.maximum_UCI_capacity
                    for hospital in self.hospitals.values()]
                    )

                current_UCIs_capacity = functools.reduce(
                    lambda x, y: x+y,
                    [hospital.current_UCI_capacity
                    for hospital in self.hospitals.values()]
                    )

                percentage_level_capacity = current_UCIs_capacity/max_UCIs_capacity

                if not self.decreed_quarantine_by_UCIs_restrictions:

                    if percentage_level_capacity <= percentage_level_to_start_quarantine:

                        # Quarantine must be decreed

                        self.decreed_quarantine = True

                        groups_in_quarantine_by_UCIs_restrictions = \
                            groups_target_to_quarantine

                        self.decreed_quarantine_by_UCIs_restrictions = True

                else:
                    # self.decreed_quarantine_by_UCIs_restrictions == True

                    if not (percentage_level_capacity > percentage_level_to_exit_quarantine):

                            # Quarantine stills being decreed

                            self.decreed_quarantine = True

                            groups_in_quarantine_by_UCIs_restrictions = \
                                groups_target_to_quarantine

                    else:
                        self.decreed_quarantine_by_UCIs_restrictions = False

                #=============================================
                # Extend self.groups_in_quarantine

                self.groups_in_quarantine.extend(groups_in_quarantine_by_UCIs_restrictions)

                self.groups_in_quarantine = list(set(self.groups_in_quarantine))

                #=============================================
                # Populate groups_in_quarantine_by_UCIs_restrictions_df

                # using fromkeys() method
                if self.decreed_quarantine_by_UCIs_restrictions:
                    quarantine_data_dict = dict.fromkeys(
                        self.groups_in_quarantine_by_UCIs_restrictions_df.columns.to_list(),
                        1
                        )
                else:
                    quarantine_data_dict = dict.fromkeys(
                        self.groups_in_quarantine_by_UCIs_restrictions_df.columns.to_list(),
                        0
                        )
                quarantine_data_dict['step'] = self.step
                quarantine_data_dict['datetime'] = self.datetime

                self.groups_in_quarantine_by_UCIs_restrictions_df = \
                    self.groups_in_quarantine_by_UCIs_restrictions_df.append(
                        quarantine_data_dict, ignore_index=True
                        )

        #=============================================
        # Restrictions by population variables
        if self.quarantine_restrictions_info[
            'restictions_by_population_variables']['enabled']:

            #=============================================
            # Variable 1: 'dead by disease'
            if self.quarantine_restrictions_info[
                'restictions_by_population_variables'][
                'conditions_for_quarantine']['dead by disease']['enabled']:

                groups_in_quarantine_by_deaths = []

                time_window = self.quarantine_restrictions_info[
                    'restictions_by_population_variables'][
                    'conditions_for_quarantine']['dead by disease']['time_window']
                
                if time_window <= self.step:

                    time_window_array = [self.step - i for i in reversed(range(time_window))]

                    percentage_level_to_start_quarantine = self.quarantine_restrictions_info[
                        'restictions_by_population_variables'][
                        'conditions_for_quarantine']['dead by disease']['percentage_level_to_start_quarantine']

                    quarantine_length = self.quarantine_restrictions_info[
                        'restictions_by_population_variables'][
                        'conditions_for_quarantine']['dead by disease']['quarantine_length']

                    groups_target_to_quarantine = self.quarantine_restrictions_info[
                        'restictions_by_population_variables'][
                        'conditions_for_quarantine']['dead by disease']['groups_target_to_quarantine']

                    window_number_of_deaths_by_disease = self.agents_info_df.loc[
                        (self.agents_info_df['step'].isin(time_window_array))
                        &
                        (self.agents_info_df['live_state'] == 'dead by disease')
                        ].shape[0]

                    window_initial_population_number = self.population_numbers[
                        time_window_array[0]
                        ]

                    percentage_of_deads = window_number_of_deaths_by_disease/window_initial_population_number

                    if not self.decreed_quarantine_by_deaths:

                        if percentage_of_deads >= percentage_level_to_start_quarantine:

                            # Quarantine must be decreed

                            self.decreed_quarantine = True

                            groups_in_quarantine_by_deaths = \
                                groups_target_to_quarantine

                            self.decreed_quarantine_by_deaths = True

                            self.time_after_decree_quarantine_by_deaths = 0

                    else:
                        # self.decreed_quarantine_by_deaths == True

                        self.time_after_decree_quarantine_by_deaths += 1

                        if self.time_after_decree_quarantine_by_deaths <= quarantine_length:

                                # Quarantine stills being decreed

                                self.decreed_quarantine = True

                                groups_in_quarantine_by_deaths = \
                                    groups_target_to_quarantine

                        else:
                            self.decreed_quarantine_by_deaths = False

                            self.time_after_decree_quarantine_by_deaths = None

                    #=============================================
                    # Extend self.groups_in_quarantine

                    self.groups_in_quarantine.extend(groups_in_quarantine_by_deaths)

                    self.groups_in_quarantine = list(set(self.groups_in_quarantine))

                    #=============================================
                    # Populate groups_in_quarantine_by_deaths_df

                    # using fromkeys() method
                    if self.decreed_quarantine_by_deaths:
                        quarantine_data_dict = dict.fromkeys(
                            self.groups_in_quarantine_by_deaths_df.columns.to_list(),
                            1
                            )
                    else:
                        quarantine_data_dict = dict.fromkeys(
                            self.groups_in_quarantine_by_deaths_df.columns.to_list(),
                            0
                            )
                    quarantine_data_dict['step'] = self.step
                    quarantine_data_dict['datetime'] = self.datetime

                    self.groups_in_quarantine_by_deaths_df = \
                        self.groups_in_quarantine_by_deaths_df.append(
                            quarantine_data_dict, ignore_index=True
                            )
                else:
                    #=============================================
                    # Populate groups_in_quarantine_by_deaths_df

                    # using fromkeys() method
                    quarantine_data_dict = dict.fromkeys(
                        self.groups_in_quarantine_by_deaths_df.columns.to_list(),
                        0
                        )
                    quarantine_data_dict['step'] = self.step
                    quarantine_data_dict['datetime'] = self.datetime

                    self.groups_in_quarantine_by_deaths_df = \
                        self.groups_in_quarantine_by_deaths_df.append(
                            quarantine_data_dict, ignore_index=True
                            )

            #=============================================
            # Variable 2: 'diagnosed'
            if self.quarantine_restrictions_info[
                'restictions_by_population_variables'][
                'conditions_for_quarantine']['diagnosed']['enabled']:

                groups_in_quarantine_by_diagnosed_people = []

                time_window = self.quarantine_restrictions_info[
                    'restictions_by_population_variables'][
                    'conditions_for_quarantine']['diagnosed']['time_window']

                if time_window <= self.step:

                    time_window_array = [self.step - i for i in reversed(range(time_window))]

                    percentage_level_to_start_quarantine = self.quarantine_restrictions_info[
                        'restictions_by_population_variables'][
                        'conditions_for_quarantine']['diagnosed']['percentage_level_to_start_quarantine']

                    quarantine_length = self.quarantine_restrictions_info[
                        'restictions_by_population_variables'][
                        'conditions_for_quarantine']['diagnosed']['quarantine_length']

                    groups_target_to_quarantine = self.quarantine_restrictions_info[
                        'restictions_by_population_variables'][
                        'conditions_for_quarantine']['diagnosed']['groups_target_to_quarantine']

                    dataframe_of_diagnosed_people = self.agents_info_df.loc[
                        (self.agents_info_df['step'].isin(time_window_array[:-1]))
                        &
                        (self.agents_info_df['live_state'] != 'dead by disease')
                        &
                        (self.agents_info_df['diagnosed'])
                        ]

                    window_diagnosed_people = dataframe_of_diagnosed_people['agent'].unique().tolist()

                    current_diagnosed_people = [
                        agent_index
                        for agent_index in self.population.keys()
                        if self.population[agent_index].diagnosed == True
                        ]
                    
                    window_diagnosed_people.extend(current_diagnosed_people)
        
                    window_total_diagnosed_people = list(set(window_diagnosed_people))

                    number_of_diagnosed_people = len(window_total_diagnosed_people)

                    window_initial_population_number = self.population_numbers[
                        time_window_array[0]
                        ]

                    percentage_of_people_diagnosed = number_of_diagnosed_people/window_initial_population_number

                    if not self.decreed_quarantine_by_diagnosed_people:

                        if percentage_of_people_diagnosed >= percentage_level_to_start_quarantine:

                            # Quarantine must be decreed

                            self.decreed_quarantine = True

                            groups_in_quarantine_by_deaths = \
                                groups_target_to_quarantine

                            self.decreed_quarantine_by_deaths = True

                            self.time_after_decree_quarantine_by_diagnosed_people = 0

                    else:
                        # self.decreed_quarantine_by_diagnosed_people == True

                        self.time_after_decree_quarantine_by_diagnosed_people += 1

                        if self.time_after_decree_quarantine_by_diagnosed_people <= quarantine_length:

                                # Quarantine stills being decreed

                                self.decreed_quarantine = True

                                groups_in_quarantine_by_diagnosed_people = \
                                    groups_target_to_quarantine

                        else:
                            self.decreed_quarantine_by_diagnosed_people = False

                    #=============================================
                    # Extend self.groups_in_quarantine

                    self.groups_in_quarantine.extend(groups_in_quarantine_by_diagnosed_people)

                    self.groups_in_quarantine = list(set(self.groups_in_quarantine))

                    #=============================================
                    # Populate groups_in_quarantine_by_diagnosed_people_df

                    # using fromkeys() method
                    if self.decreed_quarantine_by_diagnosed_people:
                        quarantine_data_dict = dict.fromkeys(
                            self.groups_in_quarantine_by_diagnosed_people_df.columns.to_list(),
                            1
                            )
                    else:
                        quarantine_data_dict = dict.fromkeys(
                            self.groups_in_quarantine_by_diagnosed_people_df.columns.to_list(),
                            0
                            )
                    quarantine_data_dict['step'] = self.step
                    quarantine_data_dict['datetime'] = self.datetime

                    self.groups_in_quarantine_by_diagnosed_people_df = \
                        self.groups_in_quarantine_by_diagnosed_people_df.append(
                            quarantine_data_dict, ignore_index=True
                            )
                else:
                    #=============================================
                    # Populate groups_in_quarantine_by_diagnosed_people_df

                    # using fromkeys() method
                    quarantine_data_dict = dict.fromkeys(
                        self.groups_in_quarantine_by_diagnosed_people_df.columns.to_list(),
                        0
                        )
                    quarantine_data_dict['step'] = self.step
                    quarantine_data_dict['datetime'] = self.datetime

                    self.groups_in_quarantine_by_diagnosed_people_df = \
                        self.groups_in_quarantine_by_diagnosed_people_df.append(
                            quarantine_data_dict, ignore_index=True
                            )

        #=============================================
        # Change population quarantine states

        # Cycle runs along all agents in current population
        if self.n_processes == 1:
            for agent_index in self.population.keys():

                self.population[agent_index].quarantine_by_government_decrees(
                    self.decreed_quarantine,
                    self.groups_in_quarantine
                    )
        else:
            pass
            # self.pool.starmap(
            #         parallel_disease_state_transition,
            #         [agent_index for agent_index in self.population.keys()]
            #     )


#===============================================================================
# In order to parallelize code

def retrieve_agent_dict(
    population,
    agent_index: int,
    step: int,
    datetime: pd._libs.tslibs.timestamps.Timestamp
    ):
    """
    """
    agent_dict = population[agent_index].getstate()
    agent_dict['step'] = step
    agent_dict['datetime'] = datetime

    return agent_dict


def spatial_trees_and_agents_indices(
    population,
    population_number: int,
    disease_state: str,
    spatial_trees_by_disease_state,
    agents_indices_by_disease_state
    ):
    """
    """
    # Spatial tree to calculate distances fastly for agents of each state
    points = [
        [population[agent_index].x, population[agent_index].y]
        for agent_index in population.keys()
        if (population[agent_index].disease_state == disease_state
            and not population[agent_index].is_hospitalized)
        ]

    agents_indices = np.array([
        agent_index
        for agent_index in population.keys()
        if (population[agent_index].disease_state == disease_state
            and not population[agent_index].is_hospitalized)
        ])

    if len(points) is not 0:
        one_percent_of_population = np.floor(population_number*0.01)
        leafsize = one_percent_of_population if one_percent_of_population > 10 else 10

        spatial_trees_by_disease_state[disease_state] = spatial.KDTree(points, leafsize=leafsize)
        agents_indices_by_disease_state[disease_state] = agents_indices
    else:
        spatial_trees_by_disease_state[disease_state] = None
        agents_indices_by_disease_state[disease_state] = None


def parallel_disease_state_transition(
    self,
    population,
    agent_index: int,
    dt: int,
    ):
    """
    """
    self.population[agent_index].disease_state_transition(self.dt)