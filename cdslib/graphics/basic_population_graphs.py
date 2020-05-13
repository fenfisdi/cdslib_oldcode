import plotly.express as px
import plotly.graph_objects as go
from cdslib.population import BasicPopulation

class BasicPopulationGraphs:
    def __init__(
        self,
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
        self.step = basic_population.step
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
        contacted_with = agent_dict['contacted_with']
        alerted_by = agent_dict['alerted_by']

        infected_info = agent_dict['infected_info']


        if live_state == 'alive':
            if len(list(infected_info.keys())) != 0:

                infected_by = agent_dict['infected_by']
                infected_in_step = agent_dict['infected_in_step']

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
                    f'<b>Contacted with</b>: {contacted_with}'
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
                    f'<b>Contacted with</b>: {contacted_with}'
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


    def plot_locations(
        self,
        step: int
        ):
        """
        """
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
                title_text='Current Population Locations',
                hovermode='closest'
                )
            )

        population_data = self.agents_info_df.loc[
            self.agents_info_df['step'] == step
            ].sort_values(by=['agent']).to_dict(orient='records')

        # Cycle runs along all agents in current population
        for agent_dict in population_data:
            # Add traces
            fig.add_trace(
                self.go_agent_scatter(agent_dict)
                )

        fig.update_layout(showlegend=False)

        fig.show()

    # TODO
    # Change velocity, frames, menu, axis labels
    def animate_population(self):
        """
        """
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

        fig.show()


    def go_line(
        self,
        x: list,
        y: list, 
        name: str
        ):
        """
        """
        states_names = list(self.disease_states_line_colors.keys())

        if name in states_names:
            return go.Scatter(
                x=x,
                y=y,
                mode='lines',
                connectgaps=False,
                line=self.disease_states_line_colors[name],
                name=name
                )
        else:
            return go.Scatter(
                x=x,
                y=y,
                mode='lines',
                connectgaps=False,
                name=name
                )


    def agents_times_series_plot(
        self,
        mode: str='states',
        column: str=None,
        inspection_values: list=None
        ):
        """
            mode: str
        """

        if mode == 'states':

            df = self.agents_info_df.loc[
                self.agents_info_df['live_state'] == 'alive' 
                ][['step', 'agent', 'disease_state']].groupby(
                    ['step', 'disease_state'],
                    as_index=False
                    ).count().copy()

            df.rename(
                columns={'agent':'agents'},
                inplace=True
                )
            
            states = self.agents_info_df['disease_state'].unique()
            
            fig = go.Figure()
            
            for state in states:
                subdf = df.loc[
                    df['disease_state'] == state
                ][['step', 'agents']].copy()

                # Add traces
                fig.add_trace(
                    self.go_line(
                        x=subdf['step'].to_list(),
                        y=subdf['agents'].to_list(),
                        name=state
                    )
                )

            fig.update_xaxes(rangeslider_visible=True)
            
            fig.update_layout(hovermode='x unified')

            fig.show()

        if mode == 'infection':
            infection_df = self.agents_info_df.copy()

            infection_df['infection_state'] = func(
                infection_df['disease_state'].to_list()
                )

            df = infection_df.loc[
                infection_df['live_state'] == 'alive' 
                ][['step', 'agent', 'infection_state']].groupby(
                    ['step', 'infection_state'],
                    as_index=False
                    ).count().copy()

            df.rename(
                columns={'agent':'agents'},
                inplace=True
                )
            
            states = infection_df['infection_state'].unique()
            
            fig = go.Figure()
            
            for state in states:
                subdf = df.loc[
                    df['infection_state'] == state
                ][['step', 'agents']].copy()

                # Add traces
                fig.add_trace(
                    self.go_line(
                        x=subdf['step'].to_list(),
                        y=subdf['agents'].to_list(),
                        name=state
                    )
                )

            fig.update_xaxes(rangeslider_visible=True)
            
            fig.update_layout(hovermode='x unified')

            fig.show()

def func(series):
    new_series = []
    for x in series:
        if x == 'susceptible':
            new_series.append(x)
        else:
            new_series.append('infected')

    return new_series