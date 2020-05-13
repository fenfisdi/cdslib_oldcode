import time
import numpy as np
import pandas as pd
import plotly.express as px

from cdslib import AgentsInfo
from cdslib import BasicPopulation
from cdslib import BasicPopulationGraphs

def time_function_1():
    Lambda = 5*6
    t = np.random.poisson(Lambda, 1)[0]
    return t if t < 21*6 else 21*6

def time_function_2():
    Lambda = 10*6
    t = np.random.poisson(Lambda, 1)[0]
    return t if t < 20*6 else 20*6

def time_function_3():
    Lambda = 14*6
    t = np.random.poisson(Lambda, 1)[0]
    return t if t < 20*6 else 20*6

def time_function_4():
    Lambda = 21*6
    t = np.random.poisson(Lambda, 1)[0]
    return t if t < 42*6 else 42*6

def time_function_5():
    Lambda = 10*6
    t = np.random.poisson(Lambda, 1)[0]
    return t if t < 15*6 else 15*6

if __name__ == "__main__":

    dt_scale_in_years = 0.000685 # 1/4 of day in years

    #===========================================================================
    # Information needed for AgentsInfo class

    age_groups = [
        'child',
        'teenager',
        'vicenarian',
        'tricenarian',
        'quadragenarian',
        'quinquagenarian',
        'sexagenarian',
        ]

    disease_states = [
        'susceptible',
        'exposed',
        'asymptomatic',
        'mildly-ill',
        'seriously-ill',
        'recovered',
        'inmune',
        'dead'
        ]

    susceptibility_groups = [
        'mildly-susceptible',
        'highly-susceptible'
        ]

    vulnerability_groups = [
        'not-vulnerable',
        'vulnerable'
        ]

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
                    'time_function': time_function_3
                    },
                'seriously-ill': {
                    'time_function': time_function_4
                    },
                'recovered': {
                    'time_function': time_function_5
                    },
                'inmune': {
                    'time_function': None
                    },
                'dead': {
                    'time_function': None
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
                    'time_function': time_function_3
                    },
                'seriously-ill': {
                    'time_function': time_function_4
                    },
                'recovered': {
                    'time_function': time_function_5
                    },
                'inmune': {
                    'time_function': None
                    },
                'dead': {
                    'time_function': None
                    }
                }
            },
        'criticality_level_of_evolution_of_disease_states': {
            'inmune': 0,
            'susceptible': 0,
            'exposed': 1,
            'asymptomatic': 2,
            'recovered': 2,
            'mildly-ill': 3,
            'seriously-ill': 4,
            'dead': 5
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
                    'becomes_into': ['inmune', 'mildly-ill', 'seriously-ill'],
                    'transition_probability': [0.75, 0.20, 0.05]
                    },
                'mildly-ill': {
                    'becomes_into': ['seriously-ill', 'recovered'],
                    'transition_probability': [0.20, 0.80]
                    },
                'seriously-ill': {
                    'becomes_into': ['recovered', 'dead'],
                    'transition_probability': [0.95, 0.05]
                    },
                'recovered': {
                    'becomes_into': ['mildly-ill', 'seriously-ill', 'inmune'],
                    'transition_probability': [0.07, 0.03, 0.90]
                    },
                'inmune': {
                    'becomes_into': None,
                    'transition_probability': None
                    },
                'dead': {
                    'becomes_into': None,
                    'transition_probability': None
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
                    'becomes_into': ['inmune', 'mildly-ill', 'seriously-ill'],
                    'transition_probability': [0.05, 0.20, 0.75]
                    },
                'mildly-ill': {
                    'becomes_into': ['seriously-ill', 'recovered'],
                    'transition_probability': [0.80, 0.20]
                    },
                'seriously-ill': {
                    'becomes_into': ['recovered', 'dead'],
                    'transition_probability': [0.20, 0.80]
                    },
                'recovered': {
                    'becomes_into': ['mildly-ill', 'seriously-ill', 'inmune'],
                    'transition_probability': [0.15, 0.15, 0.70]
                    },
                'inmune': {
                    'becomes_into': None,
                    'transition_probability': None
                    },
                'dead': {
                    'becomes_into': None,
                    'transition_probability': None
                    }
                }
            },
        'disease_states_transitions_by_illness_complications_by_vulnerability_group': {

            }
        }

    contagion_dynamics_info = {
        'contagion_probabilities_by_susceptibility_groups': {
            'mildly-susceptible': 0.5,
            'highly-susceptible': 0.9
            },
        'criticality_level_of_disease_states_to_susceptibility_to_contagion': {
            'inmune': 0,
            'susceptible': 0,
            'exposed': 1,
            'asymptomatic': 0,
            'recovered': 0,
            'mildly-ill': 0,
            'seriously-ill': 0,
            'dead': 0
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
                },
            'inmune': {
                'becomes_into': None,
                'transition_probability': None
                },
            'dead': {
                'becomes_into': None,
                'transition_probability': None
                }
            },
        'dynamics_of_disease_states_contagion': {
            'susceptible': {
                'can_get_infected': True,
                'is_infected': False,
                'can_spread': False,
                'spread_radius': None,
                'spread_probability': None
                },
            'exposed': {
                'can_get_infected': False,
                'is_infected': True,
                'can_spread': True,
                'spread_radius': 3.0,
                'spread_probability': 0.4
                },
            'asymptomatic': {
                'can_get_infected': False,
                'is_infected': True,
                'can_spread': True,
                'spread_radius': 3.0,
                'spread_probability': 0.6
                },
            'mildly-ill': {
                'can_get_infected': False,
                'is_infected': True,
                'can_spread': True,
                'spread_radius': 3.0,
                'spread_probability': 0.8
                },
            'seriously-ill': {
                'can_get_infected': False,
                'is_infected': True,
                'can_spread': True,
                'spread_radius': 3.0,
                'spread_probability': 0.8
                },
            'recovered': {
                'can_get_infected': False,
                'is_infected': True,
                'can_spread': True,
                'spread_radius': 3.0,
                'spread_probability': 0.4
                },
            'inmune': {
                'can_get_infected': False,
                'is_infected': False,
                'can_spread': False,
                'spread_radius': None,
                'spread_probability': None
                },
            'dead': {
                'can_get_infected': False,
                'is_infected': False,
                'can_spread': False,
                'spread_radius': None,
                'spread_probability': None
                }
            },
        'inmunization_level_by_vulnerability_group': {
                'not-vulnerable': 1.0,
                'vulnerable': 1.0
            }
        }

    population_age_groups_info = {
        'child': {
            'min_age': 0,
            'max_age': 9,
            'mortality_probability': dt_scale_in_years* 1.7/1000.
            },
        'teenager': {
            'min_age': 10,
            'max_age': 19,
            'mortality_probability': dt_scale_in_years* 1.4/1000.
            },
        'vicenarian': {
            'min_age': 20,
            'max_age': 29,
            'mortality_probability': dt_scale_in_years* 3.4/1000.
            },
        'tricenarian': {
            'min_age': 30,
            'max_age': 39,
            'mortality_probability': dt_scale_in_years* 3.1/1000.
            },
        'quadragenarian': {
            'min_age': 40,
            'max_age': 49,
            'mortality_probability': dt_scale_in_years* 3.7/1000.
            },
        'quinquagenarian': {
            'min_age': 50,
            'max_age': 59,
            'mortality_probability': dt_scale_in_years* 7.2/1000.
            },
        'sexagenarian': {
            'min_age': 60,
            'max_age': None,
            'mortality_probability': dt_scale_in_years* 40.9/1000.
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
                'diagnosis_time': 5*6,
                'diagnosis_probability': 0.20
                },
            'asymptomatic': {
                'can_be_diagnosed': True,
                'diagnosis_time': 5*6,
                'diagnosis_probability': 0.20
                },
            'mildly-ill': {
                'can_be_diagnosed': True,
                'diagnosis_time': 5*6,
                'diagnosis_probability': 0.10
                },
            'seriously-ill': {
                'can_be_diagnosed': True,
                'diagnosis_time': 2*6,
                'diagnosis_probability': 0.95
                },
            'recovered': {
                'can_be_diagnosed': False,
                'diagnosis_time': None,
                'diagnosis_probability': None
                },
            'inmune': {
                'can_be_diagnosed': False,
                'diagnosis_time': None,
                'diagnosis_probability': None
                },
            'dead': {
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
                'diagnosis_time': 3*6,
                'diagnosis_probability': 0.40
                },
            'asymptomatic': {
                'can_be_diagnosed': True,
                'diagnosis_time': 3*6,
                'diagnosis_probability': 0.40
                },
            'mildly-ill': {
                'can_be_diagnosed': True,
                'diagnosis_time': 3*6,
                'diagnosis_probability': 0.20
                },
            'seriously-ill': {
                'can_be_diagnosed': True,
                'diagnosis_time': 2*6,
                'diagnosis_probability': 0.95
                },
            'recovered': {
                'can_be_diagnosed': False,
                'diagnosis_time': None,
                'diagnosis_probability': None
                },
            'inmune': {
                'can_be_diagnosed': False,
                'diagnosis_time': None,
                'diagnosis_probability': None
                },
            'dead': {
                'can_be_diagnosed': False,
                'diagnosis_time': None,
                'diagnosis_probability': None
                }
            }
        }

    hospitalization_info = {
        'hospital_information': {
            'H1': {
                'x': 0.0,
                'y': 0.0,
                'hospital_capacity': 40,
                'UCI_capacity': 10
                },
            'H2': {
                'x': 0.0,
                'y': 30.0,
                'hospital_capacity': 40,
                'UCI_capacity': 20
                },
            'H3': {
                'x': -30.0,
                'y': 0.0,
                'hospital_capacity': 40,
                'UCI_capacity': 15
                }
            },
        'hospitalization_of_disease_states_by_vulnerability_group': {
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
                    },
                'inmune': {
                    'can_be_hospitalized': False,
                    'hospitalization_probability': None,
                    'UCI_probability': None
                    },
                'dead': {
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
                    },
                'inmune': {
                    'can_be_hospitalized': False,
                    'hospitalization_probability': None,
                    'UCI_probability': None
                    },
                'dead': {
                    'can_be_hospitalized': False,
                    'hospitalization_probability': None,
                    'UCI_probability': None
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
                    },
                'inmune': {
                    'should_be_alert': False,
                    'alertness_probability': None
                    },
                'dead': {
                    'should_be_alert': False,
                    'alertness_probability': None
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
                    },
                'inmune': {
                    'should_be_alert': False,
                    'alertness_probability': None
                    },
                'dead': {
                    'should_be_alert': False,
                    'alertness_probability': None
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
                    },
                'inmune': {
                    'avoidable_agent': False,
                    'avoidness_radius': None
                    },
                'dead': {
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
                    },
                'inmune': {
                    'avoidable_agent': False,
                    'avoidness_radius': None
                    },
                'dead': {
                    'avoidable_agent': False,
                    'avoidness_radius': None
                    }
                }
            }
        }

    agents_info = AgentsInfo(
        disease_states=disease_states,
        susceptibility_groups=susceptibility_groups,
        vulnerability_groups=vulnerability_groups,
        dynamics_of_the_disease_states_transitions_info= \
            dynamics_of_the_disease_states_transitions_info,
        contagion_dynamics_info=contagion_dynamics_info,
        population_age_groups_info=population_age_groups_info,
        diagnosis_of_disease_states_by_vulnerability_group= \
            diagnosis_of_disease_states_by_vulnerability_group,
        hospitalization_info= \
            hospitalization_info,
        social_distancing_info=social_distancing_info
        )

    #===========================================================================
    # Information needed for BasicPopulation class

    initial_population_number = 20

    initial_population_age_distribution = {
        'child': 0.10,
        'teenager': 0.15,
        'vicenarian': 0.20,
        'tricenarian': 0.25,
        'quadragenarian': 0.15,
        'quinquagenarian': 0.10,
        'sexagenarian': 0.05
        }

    initial_population_distributions = {
        'age_group/susceptibility_group': {
            'child': {
                'mildly-susceptible': 0.80,
                'highly-susceptible': 0.20,
                },
            'teenager': {
                'mildly-susceptible': 0.50,
                'highly-susceptible': 0.50,
                },
            'vicenarian': {
                'mildly-susceptible': 0.10,
                'highly-susceptible': 0.90,
                },
            'tricenarian': {
                'mildly-susceptible': 0.10,
                'highly-susceptible': 0.90,
                },
            'quadragenarian': {
                'mildly-susceptible': 0.10,
                'highly-susceptible': 0.90,
                },
            'quinquagenarian': {
                'mildly-susceptible': 0.10,
                'highly-susceptible': 0.90,
                },
            'sexagenarian': {
                'mildly-susceptible': 0.10,
                'highly-susceptible': 0.90,
                },
            },
        'age_group/vulnerability_group': {
            'child': {
                'not-vulnerable': 0.99,
                'vulnerable': 0.01
                },
            'teenager': {
                'not-vulnerable': 0.98,
                'vulnerable': 0.02
                },
            'vicenarian': {
                'not-vulnerable': 0.95,
                'vulnerable': 0.05
                },
            'tricenarian': {
                'not-vulnerable': 0.90,
                'vulnerable': 0.10
                },
            'quadragenarian': {
                'not-vulnerable': 0.80,
                'vulnerable': 0.20
                },
            'quinquagenarian': {
                'not-vulnerable': 0.80,
                'vulnerable': 0.20
                },
            'sexagenarian': {
                'not-vulnerable': 0.75,
                'vulnerable': 0.25
                },
            },
        'age_group/vulnerability_group/disease_state': {
            'child': {
                'not-vulnerable': {
                    'susceptible': 1.00,
                    'exposed': 0.00,
                    'asymptomatic': 0.00,
                    'mildly-ill': 0.00,
                    'seriously-ill': 0.00,
                    'recovered': 0.00,
                    'inmune': 0.00,
                    'dead': 0.00
                    },
                'vulnerable': {
                    'susceptible': 1.00,
                    'exposed': 0.00,
                    'asymptomatic': 0.00,
                    'mildly-ill': 0.00,
                    'seriously-ill': 0.00,
                    'recovered': 0.00,
                    'inmune': 0.00,
                    'dead': 0.00
                    }
                },
            'teenager': {
                'not-vulnerable': {
                    'susceptible': 0.90,
                    'exposed': 0.00,
                    'asymptomatic': 0.10,
                    'mildly-ill': 0.00,
                    'seriously-ill': 0.00,
                    'recovered': 0.00,
                    'inmune': 0.00,
                    'dead': 0.00
                    },
                'vulnerable': {
                    'susceptible': 0.90,
                    'exposed': 0.10,
                    'asymptomatic': 0.00,
                    'mildly-ill': 0.00,
                    'seriously-ill': 0.00,
                    'recovered': 0.00,
                    'inmune': 0.00,
                    'dead': 0.00
                    }
                },
            'vicenarian': {
                'not-vulnerable': {
                    'susceptible': 1.00,
                    'exposed': 0.00,
                    'asymptomatic': 0.00,
                    'mildly-ill': 0.00,
                    'seriously-ill': 0.00,
                    'recovered': 0.00,
                    'inmune': 0.00,
                    'dead': 0.00
                    },
                'vulnerable': {
                    'susceptible': 1.00,
                    'exposed': 0.00,
                    'asymptomatic': 0.00,
                    'mildly-ill': 0.00,
                    'seriously-ill': 0.00,
                    'recovered': 0.00,
                    'inmune': 0.00,
                    'dead': 0.00
                    }
                },
            'tricenarian': {
                'not-vulnerable': {
                    'susceptible': 1.00,
                    'exposed': 0.00,
                    'asymptomatic': 0.00,
                    'mildly-ill': 0.00,
                    'seriously-ill': 0.00,
                    'recovered': 0.00,
                    'inmune': 0.00,
                    'dead': 0.00
                    },
                'vulnerable': {
                    'susceptible': 1.00,
                    'exposed': 0.00,
                    'asymptomatic': 0.00,
                    'mildly-ill': 0.00,
                    'seriously-ill': 0.00,
                    'recovered': 0.00,
                    'inmune': 0.00,
                    'dead': 0.00
                    }
                },
            'quadragenarian': {
                'not-vulnerable': {
                    'susceptible': 0.80,
                    'exposed': 0.00,
                    'asymptomatic': 0.10,
                    'mildly-ill': 0.05,
                    'seriously-ill': 0.05,
                    'recovered': 0.00,
                    'inmune': 0.00,
                    'dead': 0.00
                    },
                'vulnerable': {
                    'susceptible': 0.80,
                    'exposed': 0.00,
                    'asymptomatic': 0.10,
                    'mildly-ill': 0.05,
                    'seriously-ill': 0.05,
                    'recovered': 0.00,
                    'inmune': 0.00,
                    'dead': 0.00
                    }
                },
            'quinquagenarian': {
                'not-vulnerable': {
                    'susceptible': 0.80,
                    'exposed': 0.00,
                    'asymptomatic': 0.10,
                    'mildly-ill': 0.05,
                    'seriously-ill': 0.05,
                    'recovered': 0.00,
                    'inmune': 0.00,
                    'dead': 0.00
                    },
                'vulnerable': {
                    'susceptible': 0.80,
                    'exposed': 0.00,
                    'asymptomatic': 0.10,
                    'mildly-ill': 0.05,
                    'seriously-ill': 0.05,
                    'recovered': 0.00,
                    'inmune': 0.00,
                    'dead': 0.00
                    }
                },
            'sexagenarian': {
                'not-vulnerable': {
                    'susceptible': 0.80,
                    'exposed': 0.00,
                    'asymptomatic': 0.10,
                    'mildly-ill': 0.05,
                    'seriously-ill': 0.05,
                    'recovered': 0.00,
                    'inmune': 0.00,
                    'dead': 0.00
                    },
                'vulnerable': {
                    'susceptible': 0.80,
                    'exposed': 0.00,
                    'asymptomatic': 0.10,
                    'mildly-ill': 0.05,
                    'seriously-ill': 0.05,
                    'recovered': 0.00,
                    'inmune': 0.00,
                    'dead': 0.00
                    }
                }
            },
        'inmunization_level': {
            0.00: 1.00,
            1.00: 0.00
            }
        }

    basic_population = BasicPopulation(
        age_group_list=age_groups,
        susceptibility_group_list=susceptibility_groups,
        vulnerability_group_list=vulnerability_groups,
        disease_state_list=disease_states,
        population_age_groups_info=population_age_groups_info,
        initial_population_number=initial_population_number,
        initial_population_age_distribution=initial_population_age_distribution,
        initial_population_distributions=initial_population_distributions
        )

    #===========================================================================
    # Create population

    horizontal_length = 100

    vertical_length = 100

    maximum_speed = 1.0 

    maximum_free_random_speed = 0.5

    basic_population.create_population(
        agents_info=agents_info,
        horizontal_length=horizontal_length,
        vertical_length=vertical_length,
        maximum_speed=maximum_speed,
        maximum_free_random_speed=maximum_free_random_speed,
        dt_scale_in_years=dt_scale_in_years
        )

    #===========================================================================
    # Evolve population
    start_time = time.time()

    for i in range(300):
        basic_population.evolve_population()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    
    #===========================================================================
    # Information needed for graphing

    agent_size = 10

    agent_opacity = 1.0

    agent_marker_line_width = 2

    # Markers for dead agents
    natural_death = 'hash'
    dead_by_disease = 'x-thin'
    natural_dead_color = px.colors.qualitative.Dark24[5]
    dead_color_dict = dict(
        color=natural_dead_color,
        size=agent_size,
        opacity=agent_opacity
        )

    # Group markers
    not_vulnerable_marker = 'circle'
    vulnerable_marker = 'star-diamond'

    # State colors
    susceptible_color = px.colors.qualitative.D3[2]   # dark_green
    exposed_color = px.colors.qualitative.D3[0]    # dark_blue
    asymptomatic_color = px.colors.qualitative.D3[3]     # dark_red
    mildly_ill_color = px.colors.qualitative.D3[1]  # dark_orange
    seriously_ill_color = px.colors.qualitative.G10[4] # dark_purple
    recovered_color = px.colors.qualitative.D3[7]    # dark_gray
    inmune_color = px.colors.qualitative.Dark2[5]   # dark yellow
    dead_color = px.colors.qualitative.D3[5]   # dark_brown

    vulnerability_groups_markers = {
        'not-vulnerable': not_vulnerable_marker,
        'vulnerable': vulnerable_marker
        }

    disease_states_markers_colors = {
        'susceptible': dict(
            color=susceptible_color,
            size=agent_size,
            opacity=agent_opacity
            ),
        'exposed': dict(
            color=exposed_color,
            size=agent_size,
            opacity=agent_opacity
            ),
        'asymptomatic': dict(
            color=asymptomatic_color,
            size=agent_size,
            opacity=agent_opacity
            ),
        'mildly-ill': dict(
            color=mildly_ill_color,
            size=agent_size,
            opacity=agent_opacity
            ),
        'seriously-ill': dict(
            color=seriously_ill_color,
            size=agent_size,
            opacity=agent_opacity
            ),
        'recovered': dict(
            color=recovered_color,
            size=agent_size,
            opacity=agent_opacity
            ),
        'inmune': dict(
            color=inmune_color,
            size=agent_size,
            opacity=agent_opacity
            ),
        'dead': dict(
            color=dead_color,
            size=agent_size,
            opacity=agent_opacity
            )
        }

    disease_states_line_colors = {
        'susceptible': dict(
            color=susceptible_color
            ),
        'exposed': dict(
            color=exposed_color
            ),
        'asymptomatic': dict(
            color=asymptomatic_color
            ),
        'mildly-ill': dict(
            color=mildly_ill_color
            ),
        'seriously-ill': dict(
            color=seriously_ill_color
            ),
        'recovered': dict(
            color=recovered_color
            ),
        'inmune': dict(
            color=inmune_color
            ),
        'dead': dict(
            color=dead_color
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
        disease_states_line_colors=disease_states_line_colors,
        max_length_in_px=700
        )

    #===========================================================================
    # Plot initial locations
    basic_population_graphs.plot_locations(step=0)

    #===========================================================================
    # Plot final locations
    basic_population_graphs.plot_locations(
        step=basic_population.agents_info_df['step'].to_list()[-1]
        )

    # basic_population_graphs.animate_population()

    basic_population_graphs.agents_times_series_plot()

    basic_population_graphs.agents_times_series_plot(
        mode='infection'
        )

    basic_population.agents_info_df.head()