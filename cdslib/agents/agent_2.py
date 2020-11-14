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
        immunization_level: float,
        quarantine_group: str,
        adherence_to_quarantine: float,
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
            immunization_level,
            quarantine_group,
            adherence_to_quarantine,
            diagnosed
            )

        self.fix_hidden_attributes()


    def fix_hidden_attributes(self):
        """
        """
        agent_dict = copy.deepcopy(self.__dict__)

        # Remove private attributes
        remove_list = [
            (key, value)
            for key, value in zip(agent_dict.keys(), agent_dict.values())
            if '_Agent__' in key
            ]

        for key, value in remove_list:
            new_key = key.replace('_Agent__', '_Agent2__')
            setattr(self, new_key, value)
            delattr(self, key)


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
        # Must vmax change ? ... Throw the dice
        dice = np.random.random_sample()

        if (dice <= \
            self.__vmax_change_probability_by_mobility_types[self.mobility_type]
            ):

            nested_categorical_fields = \
                vmax_constructor['nested_categorical_fields']

            nested_continous_fields = \
                vmax_constructor['nested_continous_fields']

            probability_distribution_function = \
                vmax_constructor['probability_distribution_function']

            arg_list = []

            for field in nested_categorical_fields:

                value = eval(f'self.{field}')

                arg_list.append(f"{field} = '{value}'")

            for field in nested_continous_fields:

                value = eval(f'self.{field}')

                arg_list.append(f"{field} = '{value}'")

            if arg_list:

                arg_string = ', '.join(arg_list)

                value = eval(
                        f"probability_distribution_function({arg_string})"
                        )

            else:
                value = probability_distribution_function()

            self.vmax = value