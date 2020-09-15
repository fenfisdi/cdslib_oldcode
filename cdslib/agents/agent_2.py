import copy
import numpy as np

from cdslib.hospitals import Hospital

from . import AgentsInfo, Agent


class AgentsInfo2(AgentsInfo):
    """
    """
    def __init__(
        self,
        disease_states: list,
        mobility_types: list,
        susceptibility_groups: list,
        vulnerability_groups: list,
        dynamics_of_the_disease_states_transitions_info: dict,
        contagion_dynamics_info: dict,
        population_age_groups_info: dict,
        diagnosis_of_disease_states_by_vulnerability_group: dict,
        hospitalization_of_disease_states_by_vulnerability_group: dict,
        social_distancing_info: dict,
        vmax_change_probability_by_mobility_types: dict
        ):
        """
        """
        self.mobility_types = mobility_types

        self.vmax_change_probability_by_mobility_types = \
            vmax_change_probability_by_mobility_types

        super().__init__(
            disease_states,
            susceptibility_groups,
            vulnerability_groups,
            dynamics_of_the_disease_states_transitions_info,
            contagion_dynamics_info,
            population_age_groups_info,
            diagnosis_of_disease_states_by_vulnerability_group,
            hospitalization_of_disease_states_by_vulnerability_group,
            social_distancing_info
        )


class Agent2(Agent):
    """
    """
    def __init__(
        self,
        agents_info: AgentsInfo2,
        x: float,
        y: float,
        vmax: float,
        agent: int,
        mobility_type: str,
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
        self.mobility_type: str = mobility_type

        super().__init__(
            agents_info,
            x,
            y,
            vmax,
            agent,
            disease_state,
            susceptibility_group,
            vulnerability_group,
            age,
            age_group,
            inmunization_level,
            quarantine_group,
            obedience_to_quarantine,
            diagnosed
            )


    def move(
        self,
        dt: float,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        vmax_constructor: dict
        ):
        """
        """
        if self.quarantine_state == 'Not in quarantine':

            #=============================================
            # Avoid frontier
            self.avoid_frontier(dt, xmin, xmax, ymin, ymax)

            #=============================================
            # Physical movement
            self.physical_movement(dt, vmax_constructor)


    def physical_movement(
        self,
        dt: float,
        vmax_constructor: dict
        ):
        """
        """
        if not self.is_hospitalized:
            #=============================================
            # Evolve position
            self.x = self.x + self.vx * dt
            self.y = self.y + self.vy * dt

            #=============================================
            # Update vmax

            self.update_vmax(vmax_constructor)

            #=============================================
            # Re-initialize velocity
            self.initialize_velocity()


    def update_vmax(
        self,
        vmax_constructor: dict
        ):
        """
        """
        nested_categorical_fields = \
            vmax_constructor['nested_categorical_fields']

        nested_continous_fields = \
            vmax_constructor['nested_continous_fields']

        probability_distribution_function = \
            vmax_constructor['probability_distribution_function']


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