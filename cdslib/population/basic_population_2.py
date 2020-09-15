import os
import copy
import time
import functools
import numpy as np
import pandas as pd
from scipy import spatial

from multiprocessing import Pool, Manager

from cdslib.agents import AgentsInfo2, Agent2
from cdslib.agents.agent import determine_age_group

from cdslib.hospitals import Hospital

from . import BasicPopulation

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

class BasicPopulation2(BasicPopulation):
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

        mobility_types: list,
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
        self.mobility_types: list = mobility_types
        self.vmax_constructor = initial_population_continous_data_constructor['vmax']

        super().__init__(
            hospitals_info,

            quarantine_restrictions_info,

            initial_population_number,

            population_age_groups_info,
            age_groups,
            initial_population_age_distribution,
            population_age_initialization_mode,

            susceptibility_groups,
            vulnerability_groups,
            disease_states,
            initial_population_categorical_data_distributions,
            population_categorical_data_distribution_mode,

            initial_population_continous_data_constructor,
            population_continous_data_distribution_mode,

            initial_population_categorical_data_type2_constructor,
            population_categorical_data_type2_distribution_mode,

            n_processes
            )


    def create_population(
        self,
        agents_info: AgentsInfo2,
        horizontal_length: float,
        vertical_length: float,
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

            self.population[agent_index] = Agent2(
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
        for agent_index in self.population.keys():
            self.hospitals = \
                self.population[agent_index].disease_state_transition(
                    self.dt,
                    self.hospitals,
                    self.hospitals_spatial_tree,
                    self.hospitals_labels
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
        # Retrieve population positions and velocities for agents of each state
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
                self.xmin,
                self.xmax,
                self.ymin,
                self.ymax,
                self.vmax_constructor
                )

        #=============================================
        # Populate Agents' DataFrame
        self.populate_agents_info_df()