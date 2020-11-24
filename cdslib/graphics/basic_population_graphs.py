import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from cdslib.population import BasicPopulation
from cdslib.agents import AgentsInfo
# conda install -c plotly plotly-orca==1.2.1 psutil requests

class BasicPopulationGraphs:
    def __init__(
        self,
        agents_info: AgentsInfo,
        basic_population: BasicPopulation,
        agent_marker_line_width: int,
        natural_death: str,
        dead_by_disease: str,
        dead_color_dict: dict,
        vulnerability_groups_markers: dict,
        disease_states_markers_colors: dict,
        disease_states_line_colors: dict,
        max_length_in_px: float
        ):
        """
        """
        self.agents_info_df = basic_population.agents_info_df

        self.initial_population_number = basic_population.initial_population_number

        self.disease_states = agents_info.disease_states
        self.dynamics_of_disease_states_contagion = \
            agents_info.dynamics_of_disease_states_contagion

        self.xmin = basic_population.xmin 
        self.xmax = basic_population.xmax
        self.ymin = basic_population.ymin
        self.ymax = basic_population.ymax

        # Max length
        max_length = (self.xmax - self.xmin) \
            if (self.xmax - self.xmin) > (self.ymax - self.ymin) \
            else (self.ymax - self.ymin)

        self.max_length_in_px = max_length_in_px

        self.width = self.max_length_in_px * (self.xmax - self.xmin)/max_length
        self.height = self.max_length_in_px * (self.ymax - self.ymin)/max_length

        self.agent_marker_line_width = \
            agent_marker_line_width

        self.natural_death = natural_death

        self.dead_by_disease = dead_by_disease

        self.dead_color_dict = dead_color_dict

        self.vulnerability_groups_markers = \
            vulnerability_groups_markers

        self.disease_states_markers_colors = \
            disease_states_markers_colors

        self.disease_states_line_colors = \
            disease_states_line_colors


    # TODO
    # quarantine group
    # quarantine state
    def go_agent_scatter(
        self,
        agent_dict: dict
        ):
        """
        """
        live_state = agent_dict['live_state']

        agent_label = agent_dict['agent']
        x = agent_dict['x']
        y = agent_dict['y']
        vx = agent_dict['vx']
        vy = agent_dict['vy']
        vulnerability_group = agent_dict['vulnerability_group']
        disease_state = agent_dict['disease_state']
        diagnosed = agent_dict['diagnosed']
        is_infected = agent_dict['is_infected']
        susceptible_neighbors = agent_dict['susceptible_neighbors']
        infected_spreader_neighbors = agent_dict['infected_spreader_neighbors']
        infected_non_spreader_neighbors = agent_dict['infected_non_spreader_neighbors']
        avoidable_neighbors = agent_dict['avoidable_neighbors']
        alerted_by = agent_dict['alerted_by']

        if live_state == 'alive':

            if is_infected:

                infected_by = agent_dict['infected_by']
                infected_in_step = agent_dict['infected_in_step']

                if infected_by:

                    template = (
                        f'<b>Agent</b>: {agent_label}'
                        '<br>'
                        f'<b>Position</b>: ({x:.2f}, {y:.2f})'
                        '<br>'
                        f'<b>Velocity</b>: ({vx:.2f}, {vy:.2f})'
                        '<br>'
                        f'<b>Vulnerability group</b>: {vulnerability_group}'
                        '<br>'
                        f'<b>Disease state</b>: {disease_state}'
                        '<br>'
                        f'<b>Diagnosed</b>: {diagnosed}'
                        '<br>'
                        f'<b>Infected by</b>: {infected_by}'
                        '<br>'
                        f'<b>Infected in step</b>: {infected_in_step:.0f}'
                        '<br>'
                        f'<b>Susceptible neighbors</b>: {susceptible_neighbors}'
                        '<br>'
                        f'<b>Alerted by</b>: {alerted_by}'
                        )

                else:

                    template = (
                        f'<b>Agent</b>: {agent_label}'
                        '<br>'
                        f'<b>Position</b>: ({x:.2f}, {y:.2f})'
                        '<br>'
                        f'<b>Velocity</b>: ({vx:.2f}, {vy:.2f})'
                        '<br>'
                        f'<b>Vulnerability group</b>: {vulnerability_group}'
                        '<br>'
                        f'<b>Disease state</b>: {disease_state}'
                        '<br>'
                        f'<b>Diagnosed</b>: {diagnosed}'
                        '<br>'
                        f'<b>Susceptible neighbors</b>: {susceptible_neighbors}'
                        '<br>'
                        f'<b>Alerted by</b>: {alerted_by}'
                        )


            else:
                template = (
                    f'<b>Agent</b>: {agent_label}'
                    '<br>'
                    f'<b>Position</b>: ({x:.2f}, {y:.2f})'
                    '<br>'
                    f'<b>Velocity</b>: ({vx:.2f}, {vy:.2f})'
                    '<br>'
                    f'<b>Vulnerability group</b>: {vulnerability_group}'
                    '<br>'
                    f'<b>Disease state</b>: {disease_state}'
                    '<br>'
                    f'<b>Diagnosed</b>: {diagnosed}'
                    '<br>'
                    f'<b>Infected spreader neighbors</b>: {infected_spreader_neighbors}'
                    '<br>'
                    f'<b>Infected non spreader neighbors</b>: {infected_non_spreader_neighbors}'
                    '<br>'
                    f'<b>Avoidable neighbors</b>: {avoidable_neighbors}'
                    '<br>'
                    f'<b>Alerted by</b>: {alerted_by}'
                    )


            return go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                marker_line_width=self.agent_marker_line_width,
                marker_symbol=self.vulnerability_groups_markers[vulnerability_group],
                marker=self.disease_states_markers_colors[disease_state],
                text=template,
                hoverinfo='text'
                )

        elif live_state == 'natural death':

            template = (
                f'<b>Agent</b>: {agent_label}'
                '<br>'
                f'<b>Position</b>: ({x:.2f}, {y:.2f})'
                '<br>'
                f'<b>Vulnerability group</b>: {vulnerability_group}'
                '<br>'
                f'<b>Disease state</b>: {disease_state}'
                '<br>'
                f'<b>Live state</b>: {live_state}'
                )

            return go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                marker_line_width=self.agent_marker_line_width,
                marker_symbol=self.natural_death,
                marker=self.dead_color_dict,
                text=template,
                hoverinfo='text'
                )

        else:
            # live_state == 'dead by disease'

            template = (
                f'<b>Agent</b>: {agent_label}'
                '<br>'
                f'<b>Position</b>: ({x:.2f}, {y:.2f})'
                '<br>'
                f'<b>Vulnerability group</b>: {vulnerability_group}'
                '<br>'
                f'<b>Disease state</b>: {disease_state}'
                '<br>'
                f'<b>Live state</b>: {live_state}'
                )

            return go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                marker_line_width=self.agent_marker_line_width,
                marker_symbol=self.dead_by_disease,
                marker=self.dead_color_dict,
                text=template,
                hoverinfo='text'
                )

    # TODO
    # Step or datetime info in title?
    def plot_locations(
        self,
        step: int,
        show_figure: bool=True,
        fig_name: str='population_location',
        fig_title: str='Population locations',
        show_step_in_title: bool=True,
        save_fig: bool=False,
        fig_path: str='.',
        fig_format='html' # str or list
        ):
        """
        """
        if save_fig:
            #=============================================
            # Validate fig_format

            valid_fig_format = \
                {'html', 'png', 'jpeg', 'webp', 'svg', 'pdf', 'eps'}

            if isinstance(fig_format, list):

                for single_fig_format in fig_format:

                    if single_fig_format not in valid_fig_format:
                        ErrorString = (
                            f'[ERROR] "fig_format" must be one of:'
                            '\n'
                            f'\t{valid_fig_format}'
                            )
                        raise ValueError(ErrorString)
            else:
                if fig_format not in valid_fig_format:
                    ErrorString = (
                        f'[ERROR] "fig_format" must be one of:'
                        '\n'
                        f'\t{valid_fig_format}'
                        )
                    raise ValueError(ErrorString)

        #=============================================
        # Plot

        # Set up plot
        fig = go.Figure(
            layout=go.Layout(
                width=self.width, 
                height=self.height,
                xaxis=dict(
                    range=[self.xmin, self.xmax],
                    autorange=False,
                    zeroline=False
                    ),
                yaxis=dict(
                    range=[self.ymin, self.ymax],
                    autorange=False,
                    zeroline=False
                    ),
                title_text=f'{fig_title} at step: {step}' \
                    if show_step_in_title else f'{fig_title}',
                hovermode='closest'
                )
            )

        # Retrieve population data at specified step
        population_data = self.agents_info_df.loc[
            self.agents_info_df['step'] == step
            ].sort_values(by=['agent']).to_dict(orient='records')

        # Add traces
        # Cycle runs along all agents in current population
        for agent_dict in population_data:
            fig.add_trace(
                self.go_agent_scatter(agent_dict)
                )

        # Disable legend
        fig.update_layout(showlegend=False)

        # Save figure
        if save_fig:
            fig_filename = os.path.join(fig_path, fig_name)

            if isinstance(fig_format, list):

                for single_fig_format in fig_format:

                    if single_fig_format == 'html':
                        fig.write_html(fig_filename + '.html')
                    else:
                        fig.write_image(fig_filename + f'.{single_fig_format}')
            else:
                if fig_format == 'html':
                    fig.write_html(fig_filename + '.html')
                else:
                    fig.write_image(fig_filename + f'.{fig_format}')

        if show_figure:
            fig.show()


    # TODO
    # Save in other video formats
    # Step or datetime info in title?
    # Change velocity, frames, menu, axis labels
    def animate_population(
        self,
        show_figure: bool=True,
        fig_name: str='animation',
        save_fig: bool=False,
        fig_format='html', # str
        fig_path: str='.'
        ):
        """
        """
        if save_fig:
            #=============================================
            # Validate fig_format

            valid_fig_format = \
                {'html'}

            if fig_format not in valid_fig_format:
                ErrorString = (
                    f'[ERROR] "fig_format" must be one of:'
                    '\n'
                    f'\t{valid_fig_format}'
                    )
                raise ValueError(ErrorString)

        #=============================================

        t_list = list(set(self.agents_info_df['step'].to_list()))

        # Copy dataframe
        population_df = self.agents_info_df.copy()

        # Retrieve dead agents
        dead_agents_df = population_df.loc[
            population_df['live_state'] != 'alive'
            ].copy()

        # Re-populate population_df filling each step with dead_agents in former steps
        for t in t_list:
            df = dead_agents_df.loc[
                dead_agents_df['step'] == t
                ]

            if df.shape[0] != 0:
                for future_t in range(t+1, t_list[-1]):
                    future_df = df.assign(step = future_t)

                    population_df = population_df.append(
                        future_df,
                        ignore_index=True
                        )

        # Fill full_data
        full_data = []
        
        for t in t_list:
            population_data = population_df.loc[
                population_df['step'] == t
                ].sort_values(by=['agent']).to_dict(orient='records')
            
            full_data.append(
                [self.go_agent_scatter(population_data[i])
                for i in range(len(population_data))]
                )

        # Create figure
        fig = go.Figure(
            
            data=full_data[0],
            
            layout=go.Layout(
                width=self.height, 
                height=self.width,
                xaxis=dict(
                    range=[self.xmin, self.xmax],
                    autorange=False,
                    zeroline=False
                    ),
                yaxis=dict(
                    range=[self.ymin, self.ymax],
                    autorange=False,
                    zeroline=False
                    ),
                title='Animation',
                hovermode='closest',
                updatemenus=[
                    dict(
                        type='buttons',
                        buttons=[
                            dict(
                                label='Play',
                                method='animate',
                                args=[None]
                                )
                            ]
                        )
                    ]
                ),
            frames=[go.Frame(data=full_data[t]) for t in t_list]
            )
        
        fig.update_layout(showlegend=False)

        # Save figure
        if save_fig:
            fig_filename = os.path.join(fig_path, fig_name)

            if fig_format == 'html':
                fig.write_html(fig_filename + '.html')

        # Show figure
        if show_figure:
            fig.show()


    def go_line(
        self,
        x: list,
        y: list,
        name: str,
        mode: str='lines',
        color=None
        ):
        """
        """
        states_names = list(self.disease_states_line_colors.keys())

        if name in states_names:
            return go.Scatter(
                x=x,
                y=y,
                mode=mode,
                connectgaps=False,
                line=self.disease_states_line_colors[name],
                name=name
                )
        else:
            if color is None:
                return go.Scatter(
                    x=x,
                    y=y,
                    mode=mode,
                    connectgaps=False,
                    name=name
                    )
            else:
                # color is not None
                return go.Scatter(
                    x=x,
                    y=y,
                    mode=mode,
                    connectgaps=False,
                    line=color,
                    name=name
                    )


    def disease_states_times_series_plot(
        self,
        y_format: str='percentage',
        time_format: str='datetime',
        show_figure: bool=True,
        fig_name: str='disease_states_times_series',
        fig_title: str='Disease states time series',
        save_fig: bool=False,
        fig_path: str='.',
        fig_format='html', # str or list 
        save_csv: bool=False,
        csv_path: str='.',
        dataframe_format: str='long'
        ):
        """
        """
        #=============================================
        # Validate y_format

        valid_y_format = {'percentage', 'number'}

        if y_format not in valid_y_format:
            ErrorString = (
                f'[ERROR] "y_format" must be one of:'
                '\n'
                f'\t{valid_y_format}'
                )
            raise ValueError(ErrorString)

        #=============================================
        # Validate time_format

        valid_time_format = {'step', 'datetime'}

        if time_format not in valid_time_format:
            ErrorString = (
                f'[ERROR] "time_format" must be one of:'
                '\n'
                f'\t{valid_time_format}'
                )
            raise ValueError(ErrorString)


        if save_fig:
            #=============================================
            # Validate fig_format

            valid_fig_format = \
                {'html', 'png', 'jpeg', 'webp', 'svg', 'pdf', 'eps'}

            if isinstance(fig_format, list):

                for single_fig_format in fig_format:

                    if single_fig_format not in valid_fig_format:
                        ErrorString = (
                            f'[ERROR] "fig_format" must be one of:'
                            '\n'
                            f'\t{valid_fig_format}'
                            )
                        raise ValueError(ErrorString)
            else:
                if fig_format not in valid_fig_format:
                    ErrorString = (
                        f'[ERROR] "fig_format" must be one of:'
                        '\n'
                        f'\t{valid_fig_format}'
                        )
                    raise ValueError(ErrorString)

        if save_csv:
            #=============================================
            # Validate dataframe_format

            valid_dataframe_format = {'long', 'wide'}

            if dataframe_format not in valid_dataframe_format:
                ErrorString = (
                    f'[ERROR] "dataframe_format" must be one of:'
                    '\n'
                    f'\t{valid_dataframe_format}'
                    )
                raise ValueError(ErrorString)

        #=============================================
        # Prepare dataframe

        disease_states_df = self.agents_info_df.copy()

        df = disease_states_df[[time_format, 'agent', 'disease_state']].groupby(
                [time_format, 'disease_state'],
                as_index=False
                ).count()

        if y_format == 'percentage':
            df.rename(
                columns={'agent': 'agents_pct'},
                inplace=True
                )

            df['agents_pct'] = df['agents_pct']/self.initial_population_number 
        else:
            # y_format == 'number'
            df.rename(
                columns={'agent': 'agents'},
                inplace=True
                )

        # Retrieve states
        states = self.agents_info_df['disease_state'].unique()

        # dataframe in long format
        long_format_df = df

        # Reshape df to wide format
        wide_format_df = pd.pivot_table(
            data=df,
            index=time_format,
            columns='disease_state',
            values='agents_pct' if y_format == 'percentage' else 'agents'
            )

        wide_format_df.columns.name = None
        wide_format_df.reset_index(inplace=True)

        # Save csv
        if save_csv:
            csv_filename = os.path.join(csv_path, fig_name)

            if dataframe_format == 'long':
                long_format_df.to_csv(csv_filename + '.csv', index=False)
            
            if dataframe_format == 'wide':
                wide_format_df.to_csv(csv_filename + '.csv', index=False)

        #=============================================
        # Plot

        # Set up plot
        fig = go.Figure(
            layout=go.Layout(
                title=fig_title,
                xaxis_title=time_format,
                yaxis_title='Percentage of agents' if y_format == 'percentage' else 'Number of agents'
                )
            )

        # Add traces
        for state in states:
            fig.add_trace(
                self.go_line(
                    x=wide_format_df[time_format].to_list(),
                    y=wide_format_df[state].to_list(),
                    name=state,
                    mode='lines' if state is not 'dead' else 'markers'
                )
            )

        # Add slider
        fig.update_xaxes(rangeslider_visible=True)
        fig.update_yaxes(autorange=True, fixedrange=False)

        # Set up hover
        fig.update_layout(hovermode='x unified')

        # Save figure
        if save_fig:
            fig_filename = os.path.join(fig_path, fig_name)

            if isinstance(fig_format, list):

                for single_fig_format in fig_format:

                    if single_fig_format == 'html':
                        fig.write_html(fig_filename + '.html')
                    else:
                        fig.write_image(fig_filename + f'.{single_fig_format}')
            else:
                if fig_format == 'html':
                    fig.write_html(fig_filename + '.html')
                else:
                    fig.write_image(fig_filename + f'.{fig_format}')

        # Show figure
        if show_figure:
            fig.show()


    def infection_times_series_plot(
        self,
        y_format: str='percentage',
        show_those_infected_diagnosed: bool=True,
        time_format: str='datetime',
        show_figure: bool=True,
        fig_name: str='infection_times_series',
        fig_title: str='Infection time series',
        save_fig: bool=False,
        fig_path: str='.',
        fig_format='html', # str or list
        save_csv: bool=False,
        csv_path: str='.',
        dataframe_format: str='long'
        ):
        """
        """
        #=============================================
        # Validate y_format

        valid_y_format = {'percentage', 'number'}

        if y_format not in valid_y_format:
            ErrorString = (
                f'[ERROR] "y_format" must be one of:'
                '\n'
                f'\t{valid_y_format}'
                )
            raise ValueError(ErrorString)

        #=============================================
        # Validate time_format

        valid_time_format = {'step', 'datetime'}

        if time_format not in valid_time_format:
            ErrorString = (
                f'[ERROR] "time_format" must be one of:'
                '\n'
                f'\t{valid_time_format}'
                )
            raise ValueError(ErrorString)

        if save_fig:
            #=============================================
            # Validate fig_format

            valid_fig_format = \
                {'html', 'png', 'jpeg', 'webp', 'svg', 'pdf', 'eps'}

            if isinstance(fig_format, list):

                for single_fig_format in fig_format:

                    if single_fig_format not in valid_fig_format:
                        ErrorString = (
                            f'[ERROR] "fig_format" must be one of:'
                            '\n'
                            f'\t{valid_fig_format}'
                            )
                        raise ValueError(ErrorString)
            else:
                if fig_format not in valid_fig_format:
                    ErrorString = (
                        f'[ERROR] "fig_format" must be one of:'
                        '\n'
                        f'\t{valid_fig_format}'
                        )
                    raise ValueError(ErrorString)

        if save_csv:
            #=============================================
            # Validate dataframe_format

            valid_dataframe_format = {'long', 'wide'}

            if dataframe_format not in valid_dataframe_format:
                ErrorString = (
                    f'[ERROR] "dataframe_format" must be one of:'
                    '\n'
                    f'\t{valid_dataframe_format}'
                    )
                raise ValueError(ErrorString)

        #=============================================
        # Prepare dataframe

        infection_df = self.agents_info_df.copy()

        infection_df['infection_state'] = infection_df.apply(
            lambda row: row['disease_state'] \
                if not row['is_infected'] else 'infected',
            axis=1
            )

        if not show_those_infected_diagnosed:

            df = infection_df[[time_format, 'agent', 'infection_state']].groupby(
                    [time_format, 'infection_state'],
                    as_index=False
                    ).count()

        else:
            # Show those infected diagnosed
            prev_df_1 = infection_df.loc[
                (infection_df['diagnosed'] == True)
                &
                (infection_df['infection_state'] == 'infected')
            ][[time_format, 'infection_state', 'agent']]

            prev_df_1.replace(
                to_replace='infected',
                value='diagnosed',
                inplace=True
                )

            prev_df_2 = infection_df[[time_format, 'infection_state', 'agent']]

            prev_df = pd.concat([prev_df_1, prev_df_2], ignore_index=True)

            df = prev_df.groupby(
                    [time_format, 'infection_state'],
                    as_index=False
                    ).count()

        if y_format == 'percentage':
            df.rename(
                columns={'agent': 'agents_pct'},
                inplace=True
                )

            df['agents_pct'] = df['agents_pct']/self.initial_population_number 
        else:
            # y_format == 'number'
            df.rename(
                columns={'agent': 'agents'},
                inplace=True
                )

        # Retrieve states
        states = df['infection_state'].unique()

        # dataframe in long format
        long_format_df = df

        # Reshape df to wide format
        wide_format_df = pd.pivot_table(
            data=df,
            index=time_format,
            columns='infection_state',
            values='agents_pct' if y_format == 'percentage' else 'agents'
            )

        wide_format_df.columns.name = None
        wide_format_df.reset_index(inplace=True)

        # Save csv
        if save_csv:
            csv_filename = os.path.join(csv_path, fig_name)

            if dataframe_format == 'long':
                long_format_df.to_csv(csv_filename + '.csv', index=False)
            
            if dataframe_format == 'wide':
                wide_format_df.to_csv(csv_filename + '.csv', index=False)

        #=============================================
        # Plot

        # Set up plot
        fig = go.Figure(
            layout=go.Layout(
                title=fig_title,
                xaxis_title=time_format,
                yaxis_title='Percentage of agents' if y_format == 'percentage' else 'Number of agents'
                )
            )

        # Add traces
        for state in states:
            fig.add_trace(
                self.go_line(
                    x=wide_format_df[time_format].to_list(),
                    y=wide_format_df[state].to_list(),
                    name=state,
                    mode='lines' if state is not 'dead' else 'markers'
                )
            )

        # Add slider
        fig.update_xaxes(rangeslider_visible=True)
        fig.update_yaxes(autorange=True, fixedrange=False)

        # Set up hover
        fig.update_layout(hovermode='x unified')

        # Save figure
        if save_fig:
            fig_filename = os.path.join(fig_path, fig_name)

            if isinstance(fig_format, list):

                for single_fig_format in fig_format:

                    if single_fig_format == 'html':
                        fig.write_html(fig_filename + '.html')
                    else:
                        fig.write_image(fig_filename + f'.{single_fig_format}')
            else:
                if fig_format == 'html':
                    fig.write_html(fig_filename + '.html')
                else:
                    fig.write_image(fig_filename + f'.{fig_format}')

        # Show figure
        if show_figure:
            fig.show()


    def joined_states_times_series_plot(
        self,
        joined_states: dict,
        joined_states_colors: dict,
        y_format: str='percentage',
        # show_those_infected_diagnosed: bool=True, ? TODO
        time_format: str='datetime',
        show_figure: bool=True,
        fig_name: str='joined_states_times_series',
        fig_title: str='Joined states time series',
        save_fig: bool=False,
        fig_path: str='.',
        fig_format='html', # str or list
        save_csv: bool=False,
        csv_path: str='.',
        dataframe_format: str='long'
        ):
        """
        """
        #=============================================
        # Validate y_format

        valid_y_format = {'percentage', 'number'}

        if y_format not in valid_y_format:
            ErrorString = (
                f'[ERROR] "y_format" must be one of:'
                '\n'
                f'\t{valid_y_format}'
                )
            raise ValueError(ErrorString)

        #=============================================
        # Validate time_format

        valid_time_format = {'step', 'datetime'}

        if time_format not in valid_time_format:
            ErrorString = (
                f'[ERROR] "time_format" must be one of:'
                '\n'
                f'\t{valid_time_format}'
                )
            raise ValueError(ErrorString)

        if save_fig:
            #=============================================
            # Validate fig_format

            valid_fig_format = \
                {'html', 'png', 'jpeg', 'webp', 'svg', 'pdf', 'eps'}

            if isinstance(fig_format, list):

                for single_fig_format in fig_format:

                    if single_fig_format not in valid_fig_format:
                        ErrorString = (
                            f'[ERROR] "fig_format" must be one of:'
                            '\n'
                            f'\t{valid_fig_format}'
                            )
                        raise ValueError(ErrorString)
            else:
                if fig_format not in valid_fig_format:
                    ErrorString = (
                        f'[ERROR] "fig_format" must be one of:'
                        '\n'
                        f'\t{valid_fig_format}'
                        )
                    raise ValueError(ErrorString)

        if save_csv:
            #=============================================
            # Validate dataframe_format

            valid_dataframe_format = {'long', 'wide'}

            if dataframe_format not in valid_dataframe_format:
                ErrorString = (
                    f'[ERROR] "dataframe_format" must be one of:'
                    '\n'
                    f'\t{valid_dataframe_format}'
                    )
                raise ValueError(ErrorString)

        #=============================================
        # Prepare dataframe

        joined_states_df = self.agents_info_df.copy()

        joined_states_df['state'] = joined_states_df['disease_state'].apply(
            lambda x: joined_states[x]
            )

        df = joined_states_df[[time_format, 'agent', 'state']].groupby(
                [time_format, 'state'],
                as_index=False
                ).count()

        if y_format == 'percentage':
            df.rename(
                columns={'agent': 'agents_pct'},
                inplace=True
                )

            df['agents_pct'] = df['agents_pct']/self.initial_population_number 
        else:
            # y_format == 'number'
            df.rename(
                columns={'agent': 'agents'},
                inplace=True
                )

        # Retrieve states
        states = df['state'].unique()

        # dataframe in long format
        long_format_df = df

        # Reshape df to wide format
        wide_format_df = pd.pivot_table(
            data=df,
            index=time_format,
            columns='state',
            values='agents_pct' if y_format == 'percentage' else 'agents'
            )

        wide_format_df.columns.name = None
        wide_format_df.reset_index(inplace=True)

        # Save csv
        if save_csv:
            csv_filename = os.path.join(csv_path, fig_name)

            if dataframe_format == 'long':
                long_format_df.to_csv(csv_filename + '.csv', index=False)
            
            if dataframe_format == 'wide':
                wide_format_df.to_csv(csv_filename + '.csv', index=False)

        #=============================================
        # Plot

        # Set up plot
        fig = go.Figure(
            layout=go.Layout(
                title=fig_title,
                xaxis_title=time_format,
                yaxis_title='Percentage of agents' if y_format == 'percentage' else 'Number of agents'
                )
            )

        # Add traces
        for state in states:
            fig.add_trace(
                self.go_line(
                    x=wide_format_df[time_format].to_list(),
                    y=wide_format_df[state].to_list(),
                    name=state,
                    mode='lines' if state is not 'dead' else 'markers',
                    color=joined_states_colors[state]
                )
            )

        # Add slider
        fig.update_xaxes(rangeslider_visible=True)
        fig.update_yaxes(autorange=True, fixedrange=False)

        # Set up hover
        fig.update_layout(hovermode='x unified')

        # Save figure
        if save_fig:
            fig_filename = os.path.join(fig_path, fig_name)

            if isinstance(fig_format, list):

                for single_fig_format in fig_format:

                    if single_fig_format == 'html':
                        fig.write_html(fig_filename + '.html')
                    else:
                        fig.write_image(fig_filename + f'.{single_fig_format}')
            else:
                if fig_format == 'html':
                    fig.write_html(fig_filename + '.html')
                else:
                    fig.write_image(fig_filename + f'.{fig_format}')

        # Show figure
        if show_figure:
            fig.show()