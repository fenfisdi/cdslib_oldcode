import time
import numpy as np
import pandas as pd
import plotly.express as px

from cdslib import AgentsInfo
from cdslib import BasicPopulation
from cdslib import BasicPopulationGraphs

def time_function_1():
    Lambda = 5
    return np.random.poisson(Lambda, 1)[0]

def time_function_2():
    mu = 15
    sigma = 10
    return np.random.normal(mu, sigma, 1)[0]

def time_function_3():
    Lambda = 10
    return np.random.poisson(Lambda, 1)[0]

def initial_population_group_state_scene(
    population_number,
    vulnerable_number
    ):
    population_group_list = [
        vulnerability_groups[1] if i < vulnerable_number else vulnerability_groups[0]
        for i in range(population_number)
    ]
    population_state_list = [
        disease_states[1] if i%3 == 0 else disease_states[0]
        for i in range(population_number)
    ]
    
    return population_group_list, population_state_list

if __name__ == "__main__":

    population_number = 100

    vulnerable_population_pct = 0.2

    vulnerable_number = np.ceil(population_number * vulnerable_population_pct)

    horizontal_length = 100

    vertical_length = 100

    maximum_speed = 1.0 

    maximum_free_random_speed = 0.5

    dt_scale_in_years = 0.000685 # 1/4 of day in years

    #===========================================================================
    # Information needed for Agents_info class

    disease_states = [
        'susceptible',
        'exposed',
        'asymptomatic',
        'mildly-ill',
        'seriously-ill',
        'recovered'
        ]

    vulnerability_groups = [
        'not-vulnerable',
        'vulnerable'
        ]

    population_age_groups_info = {
        'child': {
            'min_age': 0,
            'max_age': 9,
            'mortality_probability': 0.003
            },
        'teenager': {
            'min_age': 10,
            'max_age': 19,
            'mortality_probability': 0.002
            },
        'vicenarian': {
            'min_age': 20,
            'max_age': 29,
            'mortality_probability': 0.002
            },
        'tricenarian': {
            'min_age': 30,
            'max_age': 39,
            'mortality_probability': 0.004
            },
        'quadragenarian': {
            'min_age': 40,
            'max_age': 49,
            'mortality_probability': 0.004
            },
        'quinquagenarian': {
            'min_age': 50,
            'max_age': 59,
            'mortality_probability': 0.006
            },
        'sexagenarian': {
            'min_age': 60,
            'max_age': 69,
            'mortality_probability': 0.010
            },
        'septuagenarian': {
            'min_age': 70,
            'max_age': 79,
            'mortality_probability': 0.040
            },
        'octogenarian': {
            'min_age': 80,
            'max_age': 89,
            'mortality_probability': 0.070
            },
        'nonagenarian': {
            'min_age': 90,
            'max_age': 99,
            'mortality_probability': 0.095
            },
        'centenarian': {
            'min_age': 100,
            'max_age': 200,
            'mortality_probability': 0.095
            }
        }

    mortality_of_disease_states_by_vulnerability_group = {
            'not-vulnerable': {
                'susceptible': {
                    'mortality_probability': None
                    },
                'exposed': {
                    'mortality_probability': None
                    },
                'asymptomatic': {
                    'mortality_probability': None
                    },
                'mildly-ill': {
                    'mortality_probability': None
                    },
                'seriously-ill': {
                    'mortality_probability': 0.10
                    },
                'recovered': {
                    'mortality_probability': None
                    }
                },
            'vulnerable': {
                'susceptible': {
                    'mortality_probability': None
                    },
                'exposed': {
                    'mortality_probability': None
                    },
                'asymptomatic': {
                    'mortality_probability': None
                    },
                'mildly-ill': {
                    'mortality_probability': None
                    },
                'seriously-ill': {
                    'mortality_probability': 0.90
                    },
                'recovered': {
                    'mortality_probability': None
                    }
                }
            }

    # Here we are trying to model the probability that any agent is taken into
    # account for diagnosis according to its state and vulnerability group
    diagnosis_of_disease_states_by_vulnerability_group = {
            'not-vulnerable': {
                'susceptible': {
                    'can_be_diagnosed': False,
                    'diagnosis_time': None,
                    'diagnosis_probability': None
                    },
                'exposed': {
                    'can_be_diagnosed': True,
                    'diagnosis_time': 5,
                    'diagnosis_probability': 0.20
                    },
                'asymptomatic': {
                    'can_be_diagnosed': True,
                    'diagnosis_time': 5,
                    'diagnosis_probability': 0.20
                    },
                'mildly-ill': {
                    'can_be_diagnosed': True,
                    'diagnosis_time': 5,
                    'diagnosis_probability': 0.10
                    },
                'seriously-ill': {
                    'can_be_diagnosed': True,
                    'diagnosis_time': 2,
                    'diagnosis_probability': 0.95
                    },
                'recovered': {
                    'can_be_diagnosed': False,
                    'diagnosis_time': None,
                    'diagnosis_probability': None
                    }
                },
            'vulnerable': {
                'susceptible': {
                    'can_be_diagnosed': False,
                    'diagnosis_time': None,
                    'diagnosis_probability': None
                    },
                'exposed': {
                    'can_be_diagnosed': True,
                    'diagnosis_time': 3,
                    'diagnosis_probability': 0.40
                    },
                'asymptomatic': {
                    'can_be_diagnosed': True,
                    'diagnosis_time': 3,
                    'diagnosis_probability': 0.40
                    },
                'mildly-ill': {
                    'can_be_diagnosed': True,
                    'diagnosis_time': 3,
                    'diagnosis_probability': 0.20
                    },
                'seriously-ill': {
                    'can_be_diagnosed': True,
                    'diagnosis_time': 2,
                    'diagnosis_probability': 0.95
                    },
                'recovered': {
                    'can_be_diagnosed': False,
                    'diagnosis_time': None,
                    'diagnosis_probability': None
                    }
                }
            }

    hospitalization_of_disease_states_by_vulnerability_group = {
            'not-vulnerable': {
                'susceptible': {
                    'can_be_hospitalized': False,
                    'hospitalization_probability': None,
                    'UCI_probability': None
                    },
                'exposed': {
                    'can_be_hospitalized': False,
                    'hospitalization_probability': None,
                    'UCI_probability': None
                    },
                'asymptomatic': {
                    'can_be_hospitalized': False,
                    'hospitalization_probability': None,
                    'UCI_probability': None
                    },
                'mildly-ill': {
                    'can_be_hospitalized': True,
                    'hospitalization_probability': 0.05,
                    'UCI_probability': 0.0
                    },
                'seriously-ill': {
                    'can_be_hospitalized': True,
                    'hospitalization_probability': 1.0,
                    'UCI_probability': 0.10
                    },
                'recovered': {
                    'can_be_hospitalized': False,
                    'hospitalization_probability': None,
                    'UCI_probability': None
                    }
                },
            'vulnerable': {
                'susceptible': {
                    'can_be_hospitalized': False,
                    'hospitalization_probability': None,
                    'UCI_probability': None
                    },
                'exposed': {
                    'can_be_hospitalized': False,
                    'hospitalization_probability': None,
                    'UCI_probability': None
                    },
                'asymptomatic': {
                    'can_be_hospitalized': False,
                    'hospitalization_probability': None,
                    'UCI_probability': None
                    },
                'mildly-ill': {
                    'can_be_hospitalized': True,
                    'hospitalization_probability': 0.30,
                    'UCI_probability': 0.0
                    },
                'seriously-ill': {
                    'can_be_hospitalized': True,
                    'hospitalization_probability': 1.0,
                    'UCI_probability': 0.50
                    },
                'recovered': {
                    'can_be_hospitalized': False,
                    'hospitalization_probability': None,
                    'UCI_probability': None
                    }
                }
            }

    agents_info = AgentsInfo(
        disease_states=disease_states,
        vulnerability_groups=vulnerability_groups,
        population_age_groups_info=population_age_groups_info,
        mortality_of_disease_states_by_vulnerability_group= \
            mortality_of_disease_states_by_vulnerability_group,
        diagnosis_of_disease_states_by_vulnerability_group= \
            diagnosis_of_disease_states_by_vulnerability_group,
        hospitalization_of_disease_states_by_vulnerability_group= \
            hospitalization_of_disease_states_by_vulnerability_group
    )

    #===========================================================================
    # Information needed for BasicPopulation class

    contagion_dynamics_info = {
        'contagion_probabilities_by_susceptibility_groups': {
            'not-vulnerable': 0.5,
            'vulnerable': 0.9
            },
        'disease_states_transitions_by_contagion': {
            'susceptible': {
                'becomes_into': ['exposed'],
                'transition_probability': [1.0]
                },
            'exposed': {
                'becomes_into': None,
                'transition_probability': None
                },
            'asymptomatic': {
                'becomes_into': None,
                'transition_probability': None
                },
            'mildly-ill': {
                'becomes_into': None,
                'transition_probability': None
                },
            'seriously-ill': {
                'becomes_into': None,
                'transition_probability': None
                },
            'recovered': {
                'becomes_into': None,
                'transition_probability': None
                }
            },
        'dynamics_of_disease_states_contagion': {
            'susceptible': {
                'can_get_infected': True,
                'can_spread': False,
                'spread_radius': None,
                'spread_probability': None
                },
            'exposed': {
                'can_get_infected': False,
                'can_spread': True,
                'spread_radius': 3.0,
                'spread_probability': 0.4
                },
            'asymptomatic': {
                'can_get_infected': False,
                'can_spread': True,
                'spread_radius': 3.0,
                'spread_probability': 0.6
                },
            'mildly-ill': {
                'can_get_infected': False,
                'can_spread': True,
                'spread_radius': 3.0,
                'spread_probability': 0.8
                },
            'seriously-ill': {
                'can_get_infected': False,
                'can_spread': True,
                'spread_radius': 3.0,
                'spread_probability': 0.8
                },
            'recovered': {
                'can_get_infected': False,
                'can_spread': True,
                'spread_radius': 3.0,
                'spread_probability': 0.4
                }
            },
        'inmunization_level_info': {

            }
        }

    dynamics_of_the_disease_states_transitions_info = {
        'disease_states_time_functions': {
            'not-vulnerable': {
                'susceptible': {
                    'time_function': None
                    },
                'exposed': {
                    'time_function': time_function_1
                    },
                'asymptomatic': {
                    'time_function': time_function_2
                    },
                'mildly-ill': {
                    'time_function': time_function_2
                    },
                'seriously-ill': {
                    'time_function': time_function_2
                    },
                'recovered': {
                    'time_function': time_function_3
                    }
                },
            'vulnerable': {
                'susceptible': {
                    'time_function': None
                    },
                'exposed': {
                    'time_function': time_function_1
                    },
                'asymptomatic': {
                    'time_function': time_function_2
                    },
                'mildly-ill': {
                    'time_function': time_function_2
                    },
                'seriously-ill': {
                    'time_function': time_function_2
                    },
                'recovered': {
                    'time_function': time_function_3
                    }
                }
            },
        'disease_states_transitions_by_vulnerability_group': {
            'not-vulnerable': {
                'susceptible': {
                    'becomes_into': None,
                    'transition_probability': None
                    },
                'exposed': {
                    'becomes_into': ['asymptomatic', 'mildly-ill', 'seriously-ill'],
                    'transition_probability': [0.60, 0.25, 0.15]
                    },
                'asymptomatic': {
                    'becomes_into': ['susceptible', 'mildly-ill', 'seriously-ill'],
                    'transition_probability': [0.75, 0.20, 0.05]
                    },
                'mildly-ill': {
                    'becomes_into': ['seriously-ill', 'recovered'],
                    'transition_probability': [0.20, 0.80]
                    },
                'seriously-ill': {
                    'becomes_into': ['recovered'],
                    'transition_probability': [0.95]
                    },
                'recovered': {
                    'becomes_into': ['susceptible', 'mildly-ill', 'seriously-ill'],
                    'transition_probability': [0.90, 0.07, 0.03]
                    }
                },
            'vulnerable': {
                'susceptible': {
                    'becomes_into': None,
                    'transition_probability': None
                    },
                'exposed': {
                    'becomes_into': ['asymptomatic', 'mildly-ill', 'seriously-ill'],
                    'transition_probability': [0.15, 0.25, 0.60]
                    },
                'asymptomatic': {
                    'becomes_into': ['susceptible', 'mildly-ill', 'seriously-ill'],
                    'transition_probability': [0.05, 0.20, 0.75]
                    },
                'mildly-ill': {
                    'becomes_into': ['seriously-ill', 'recovered'],
                    'transition_probability': [0.80, 0.20]
                    },
                'seriously-ill': {
                    'becomes_into': ['recovered'],
                    'transition_probability': [0.20]
                    },
                'recovered': {
                    'becomes_into': ['susceptible', 'mildly-ill', 'seriously-ill'],
                    'transition_probability': [0.70, 0.15, 0.15]
                    }
                }
            }
        }

    social_distancing_info = {
        # Here we are trying to model if an agent should be alert
        # according to its vulnerability group and state
        # Note that alertness depends on a probability, which tries to model
        # the probability that an agent with a defined group and state is alert
        'dynamics_of_alertness_of_disease_states_by_vulnerability_group': {
            'not-vulnerable': {
                'susceptible': {
                    'should_be_alert': False,
                    'alertness_probability': None
                    },
                'exposed': {
                    'should_be_alert': False,
                    'alertness_probability': None
                    },
                'asymptomatic': {
                    'should_be_alert': False,
                    'alertness_probability': None
                    },
                'mildly-ill': {
                    'should_be_alert': True,
                    'alertness_probability': 0.20
                    },
                'seriously-ill': {
                    'should_be_alert': False,
                    'alertness_probability': None
                    },
                'recovered': {
                    'should_be_alert': True,
                    'alertness_probability': 0.80
                    }
                },
            'vulnerable': {
                'susceptible': {
                    'should_be_alert': True,
                    'alertness_probability': 0.80
                    },
                'exposed': {
                    'should_be_alert': True,
                    'alertness_probability': 0.80
                    },
                'asymptomatic': {
                    'should_be_alert': True,
                    'alertness_probability': 0.80
                    },
                'mildly-ill': {
                    'should_be_alert': True,
                    'alertness_probability': 0.80
                    },
                'seriously-ill': {
                    'should_be_alert': True,
                    'alertness_probability': 0.90
                    },
                'recovered': {
                    'should_be_alert': True,
                    'alertness_probability': 1.0
                    }
                }
            },
        # Here we are trying to model social dynamics of avoidness
        # about other agents
        # Note that an 'avoidable_agent' depends on its state but also on each group,
        # it means that each group defines which state is avoidable
        'dynamics_of_avoidance_of_disease_states_by_vulnerability_group': {
            'not-vulnerable': {
                'susceptible': {
                    'avoidable_agent': False,
                    'avoidness_radius': None
                    },
                'exposed': {
                    'avoidable_agent': False,
                    'avoidness_radius': None
                    },
                'asymptomatic': {
                    'avoidable_agent': False,
                    'avoidness_radius': None
                    },
                'mildly-ill': {
                    'avoidable_agent': True,
                    'avoidness_radius': 1.0
                    },
                'seriously-ill': {
                    'avoidable_agent': True,
                    'avoidness_radius': 3.0
                    },
                'recovered': {
                    'avoidable_agent': False,
                    'avoidness_radius': None
                    }
                },
            'vulnerable': {
                'susceptible': {
                    'avoidable_agent': True,
                    'avoidness_radius': 1.0
                    },
                'exposed': {
                    'avoidable_agent': True,
                    'avoidness_radius': 1.0
                    },
                'asymptomatic': {
                    'avoidable_agent': True,
                    'avoidness_radius': 1.0
                    },
                'mildly-ill': {
                    'avoidable_agent': True,
                    'avoidness_radius': 2.0
                    },
                'seriously-ill': {
                    'avoidable_agent': True,
                    'avoidness_radius': 3.0
                    },
                'recovered': {
                    'avoidable_agent': True,
                    'avoidness_radius': 1.0
                    }
                }
            }
        }

    basic_population = BasicPopulation(
        disease_states=disease_states,
        vulnerability_groups=vulnerability_groups,
        contagion_dynamics_info=contagion_dynamics_info,
        dynamics_of_the_disease_states_transitions_info= \
            dynamics_of_the_disease_states_transitions_info,
        social_distancing_info=social_distancing_info
        )

    population_group_list, population_state_list = initial_population_group_state_scene(
        population_number,
        vulnerable_number
        )

    basic_population.create_population(
        agents_info=agents_info,
        population_number=population_number,
        vulnerable_number=vulnerable_number,
        population_group_list=population_group_list,
        population_state_list=population_state_list,
        horizontal_length=horizontal_length,
        vertical_length=vertical_length,
        maximum_speed=maximum_speed,
        maximum_free_random_speed=maximum_free_random_speed,
        dt_scale_in_years=dt_scale_in_years
        )
    
    #===========================================================================
    # Information needed for graphing

    agent_size = 10

    agent_opacity = 1.0

    agent_marker_line_width = 2

    # Markers for dead agents
    natural_death = 'hash'
    dead_by_disease = 'x-thin'
    dead_color = px.colors.qualitative.Dark24[5]
    dead_color_dict = dict(
        color=dead_color,
        size=agent_size,
        opacity=agent_opacity
        )

    # Group markers
    not_vulnerable_marker = 'circle'
    vulnerable_marker = 'star-diamond'

    # State colors
    dark_green = px.colors.qualitative.D3[2]   # susceptible
    dark_blue = px.colors.qualitative.D3[0]    # incubator
    dark_red = px.colors.qualitative.D3[3]     # seriously-ill
    dark_orange = px.colors.qualitative.D3[1]  # mildly-ill
    dark_purple = px.colors.qualitative.G10[4] # asymptomatic
    dark_gray = px.colors.qualitative.D3[7]    # recovered
    dark_brown = px.colors.qualitative.D3[5]   # allegedly-recovered

    vulnerability_groups_markers = {
        'not-vulnerable': not_vulnerable_marker,
        'vulnerable': vulnerable_marker
        }

    disease_states_markers_colors = {
        'susceptible': dict(
            color=dark_green,
            size=agent_size,
            opacity=agent_opacity
            ),
        'exposed': dict(
            color=dark_blue,
            size=agent_size,
            opacity=agent_opacity
            ),
        'asymptomatic': dict(
            color=dark_purple,
            size=agent_size,
            opacity=agent_opacity
            ),
        'mildly-ill': dict(
            color=dark_orange,
            size=agent_size,
            opacity=agent_opacity
            ),
        'seriously-ill': dict(
            color=dark_red,
            size=agent_size,
            opacity=agent_opacity
            ),
        'recovered': dict(
            color=dark_gray,
            size=agent_size,
            opacity=agent_opacity
            )
        }

    disease_states_line_colors = {
        'susceptible': dict(
            color=dark_green
            ),
        'exposed': dict(
            color=dark_blue
            ),
        'asymptomatic': dict(
            color=dark_purple
            ),
        'mildly-ill': dict(
            color=dark_orange
            ),
        'seriously-ill': dict(
            color=dark_red
            ),
        'recovered': dict(
            color=dark_gray
            )
        }

    basic_population_graphs = BasicPopulationGraphs(
        basic_population=basic_population,
        agent_marker_line_width=agent_marker_line_width,
        natural_death=natural_death,
        dead_by_disease=dead_by_disease,
        dead_color_dict=dead_color_dict,
        vulnerability_groups_markers=vulnerability_groups_markers,
        disease_states_markers_colors=disease_states_markers_colors,
        disease_states_line_colors=disease_states_line_colors
        )

    #===========================================================================
    # Plot initial location
    basic_population_graphs.plot_current_locations()

    start_time = time.time()

    for i in range(100):
        basic_population.evolve_population()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    # Change plot_current_locations to plot_locations(step)
    basic_population_graphs = BasicPopulationGraphs(
        basic_population=basic_population,
        agent_marker_line_width=agent_marker_line_width,
        natural_death=natural_death,
        dead_by_disease=dead_by_disease,
        dead_color_dict=dead_color_dict,
        vulnerability_groups_markers=vulnerability_groups_markers,
        disease_states_markers_colors=disease_states_markers_colors,
        disease_states_line_colors=disease_states_line_colors
        )

    basic_population_graphs.plot_current_locations()

    #agents.animate_population()

    basic_population_graphs.agents_times_series_plot(basic_population.agents_info_df)