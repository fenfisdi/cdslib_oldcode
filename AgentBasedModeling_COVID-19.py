import os
import copy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import spatial

import plotly.express as px
import plotly.graph_objects as go

agent_size = 10

agent_opacity = 1.0

agent_marker_line_width = 2

# Markers for dead agents
natural_death = 'hash'
dead_by_disease = 'x'
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


dim = 2

population_number = 100

vulnerable_population_pct = 0.2

vulnerable_number = np.ceil(population_number * vulnerable_population_pct)

horizontal_length = 100

vertical_length = 100

maximum_speed = 1.0 

maximum_free_random_speed = 0.5

dt_scale_in_years = 0.000685 # 1/4 of day in years

# For init
population_possible_states = [
    'susceptible',
    'exposed',
    'asymptomatic',
    'mildly-ill',
    'seriously-ill',
    'recovered'
]

population_possible_groups = ['not-vulnerable', 'vulnerable']

# For coding
population_possible_groups_markers = {
    'not-vulnerable': not_vulnerable_marker,
    'vulnerable': vulnerable_marker
}

# Here we are trying to model if an agent should be alert
# according to its vulnerability group and state
# Note that alertness depends on a probability, which tries to model
# the probability that an agent with a defined group and state is alert
state_alertness_dynamics_by_group = {
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
}

# Here we are trying to model social dynamics of avoidness
# about other agents
# Note that an 'avoidable_agent' depends on its state but also on each group,
# it means that each group defines which state is avoidable
state_avoidness_dynamics_by_group = {
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

#==============================================
contagion_states_transitions_dict = {
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
}

contagion_susceptibility_groups_probabilities = {
    'not-vulnerable': 0.5,
    'vulnerable': 0.9
}

states_contagion_dynamics = {
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
}

inmunization_dict = {

}
#==================================================

states_time_functions_dict = {
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
}

states_transitions_by_group_dict = {
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

#===============================================

states_hospitalization_by_group_dict = {
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

# Here we are trying to model the probability that any agent is taken into
# account for diagnosis according to its state and vulnerability group
states_diagnosis_by_group_dict = {
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

states_mortality_by_group_dict = {
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

population_age_groups_dict = {
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

population_color_dicts = {
    'not-vulnerable': {
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
    },
    'vulnerable': {
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
}

line_color_dicts = {
    'not-vulnerable': {
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
    },
    'vulnerable': {
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
}


#===========================================================================

class Agent():
    """
    """
    def __init__(
        self,
        agent: int,
        x: float,
        y: float,
        vmax: float,
        group: str,
        state: str,
        age: float,
        diagnosed: bool=False,
        inmunization_level: float=0.0
    ):
        """
        """
        # Define agent label
        self.agent = agent

        # Define position
        self.x = x
        self.y = y

        # Define initial velocity between (-vmax , +vmax)
        self.vx, self.vy = 2. * vmax * np.random.random_sample(dim) - vmax

        # Initialize group, state, and diagnosed state
        self.group = group

        self.state = state

        self.diagnosed = diagnosed

        self.hospitalized = False

        self.requires_UCI = False

        self.waiting_diagnosis = False

        self.time_waiting_diagnosis = None

        self.infected_by = None

        self.inmunization_level = inmunization_level

        self.alertness = False

        self.alerted_by = []

        self.state_max_time = None

        self.state_time = 0

        self.live_state = 'alive'

        self.age = age

        # Determine age group
        self.determine_age_group()


    def determine_age_group(self):
        """
        """
        for age_group in population_age_groups_dict.keys():

            age_group_dict = population_age_groups_dict[age_group]

            if (age_group_dict['min_age'] <= self.age
            and np.floor(self.age) <= age_group_dict['max_age']):

                self.age_group = age_group


    def age_and_death_verification(self, dt_scale_in_years: float):
        """
        """
        dead = False

        #=============================================
        # Increment age
        self.age += dt_scale_in_years

        # Determine age group
        self.determine_age_group()

        #=============================================
        # Verify: natural death ? ... Throw the dice
        dice = np.random.random_sample()

        if dice <= population_age_groups_dict[self.age_group]['mortality_probability']:
            # The agent died
            dead = True

            # Agent died by natural reasons
            self.live_state = 'natural death'

        #=============================================
        # Verify: dead by disease ? ... Throw the dice
        dice = np.random.random_sample()

        if states_mortality_by_group_dict[self.group][self.state]['mortality_probability']:
            if dice <= states_mortality_by_group_dict[self.group][self.state]['mortality_probability']:
                # The agent died
                dead = True

                # Agent die by disease
                self.live_state = 'dead by disease'

        return dead


    def update_diagnosis_state(self, dt: float, changed_state: bool=False):
        """
        """
        if not changed_state:
            # Here we are trying to model the probability that any agent is taken into account for diagnosis
            # according to its state and vulnerability group        

            #=============================================
            # Not diagnosed and not waiting diagnosis ?
            # Then verify if is going to be diagnosed

            if not self.diagnosed:

                if not self.waiting_diagnosis:

                    if states_diagnosis_by_group_dict[self.group][self.state]['can_be_diagnosed']:

                        # Verify: is going to be diagnosed ? ... Throw the dice
                        dice = np.random.random_sample()

                        if dice <= states_diagnosis_by_group_dict[
                            self.group][self.state]['diagnosis_probability']:

                            # Agent is going to be diagnosed !!!
                            self.waiting_diagnosis = True

                            self.time_waiting_diagnosis = 0

                else:
                    # Agent is waiting diagnosis

                    #=============================================
                    # Increment time waiting diagnosis
                    self.time_waiting_diagnosis += dt

                    if self.time_waiting_diagnosis == states_diagnosis_by_group_dict[
                        self.group][self.state]['diagnosis_time']:

                        # Can be diagnosed ? (in other words: Is infected ?)
                        if states_diagnosis_by_group_dict[
                            self.group][self.state]['can_be_diagnosed']:

                            self.diagnosed = True

                            self.waiting_diagnosis = False

                            self.time_waiting_diagnosis = None

                        else:
                            self.diagnosed = False

                            self.waiting_diagnosis = False

                            self.time_waiting_diagnosis = None

            else:
                # Agent is diagnosed
                pass
        else:
            # Agent changed state

            # If agent cannot be diagnosed
            if not states_diagnosis_by_group_dict[self.group][self.state]['can_be_diagnosed']:
                
                self.diagnosed = False

                self.waiting_diagnosis = False

                self.time_waiting_diagnosis = None


    def update_hospitalization_state(self):
        # Agent can be hospitalized ?
        if states_hospitalization_by_group_dict[
            self.group][self.state]['can_be_hospitalized']:
        
            if not self.hospitalized:

                #=============================================
                # Verify: is hospitalized ? ... Throw the dice
                dice = np.random.random_sample()

                if (states_hospitalization_by_group_dict[self.group][self.state]['hospitalization_probability']
                and dice <= states_hospitalization_by_group_dict[
                    self.group][self.state]['hospitalization_probability']):
                    
                    # Agent is hospitalized !!!
                    self.hospitalized = True

                    #=============================================
                    # Verify: needs UCI ? ... Throw the dice
                    dice = np.random.random_sample()

                    if (states_hospitalization_by_group_dict[self.group][self.state]['UCI_probability']
                    and dice <= states_hospitalization_by_group_dict[
                        self.group][self.state]['UCI_probability']):
                        
                        # Agent needs UCI !!!
                        self.requires_UCI = True

            else:
                # Agent doesn't need hospitalization neither UCI
                self.hospitalized = False

                self.requires_UCI = False
        else:
            # Agent can not be hospitalized ?
            self.hospitalized = False

            self.requires_UCI = False


    def alert_avoid_agents(self, velocities_to_avoid: list):
        """
        """
        #=============================================
        # Retrieve own velocity
        own_initial_velocity = np.array([self.vx, self.vy])
        norm_own_initial_velocity = np.sqrt(np.dot(own_initial_velocity, own_initial_velocity))

        #=============================================
        # Create a vector to save new velocity vector
        new_velocity = np.array([0., 0.], dtype='float64')

        #=============================================
        # Avoid velocities
        for velocity_to_avoid in velocities_to_avoid:

            norm_velocity_to_avoid = np.sqrt(np.dot(velocity_to_avoid, velocity_to_avoid))

            # Find angle theta between both velocities
            costheta = np.dot(own_initial_velocity, velocity_to_avoid) / (norm_own_initial_velocity*norm_velocity_to_avoid)
            costheta = 1.0 if costheta > 1.0 else (0.0 if costheta < 0.0 else costheta)

            v_parallel = own_initial_velocity * costheta
            v_perpendicular = - own_initial_velocity * np.sqrt(1.0 - costheta**2)

            #=============================================
            # Add up to new_velocity
            new_velocity += v_parallel + v_perpendicular

        #=============================================
        # Disaggregate new_velocity
        self.vx, self.vy = new_velocity


    def move(self, dt: float, maximum_free_random_speed: float, xmin: float, xmax: float, ymin: float, ymax: float):
        """
        """
        #=============================================
        # Avoid frontier
        self.avoid_frontier(dt, xmin, xmax, ymin, ymax)

        #=============================================
        # Physical movement
        self.physical_movement(dt,  maximum_free_random_speed)


    def avoid_frontier(self, dt: float, xmin: float, xmax: float, ymin: float, ymax: float):
        """
        """
        #=============================================
        # Bouncing from the walls
        new_x = self.x + self.vx * dt
        new_y = self.y + self.vy * dt

        #=============================================
        # Hitting upper or lower wall 
        if new_x > xmax or new_x < xmin:
            # self.x = self.x - self.vx * dt
            self.vx = - self.vx

        #=============================================
        # Hitting left or right wall
        if new_y > ymax or new_y < ymin:
            # self.y = self.y - self.vy * dt
            self.vy = - self.vy


    def physical_movement(self, dt: float, maximum_free_random_speed: float):
        """
        """
        #=============================================
        # Evolve position
        self.x = self.x + self.vx * dt
        self.y = self.y + self.vy * dt


        if not self.alertness:
            #=============================================
            # Evolve velocity as a random walk
            dvx = maximum_free_random_speed * np.random.random_sample() - maximum_free_random_speed/2
            self.vx = self.vx + dvx

            dvy = maximum_free_random_speed * np.random.random_sample() - maximum_free_random_speed/2
            self.vy = self.vy + dvy



#===========================================================================

# TODO
# inmunization_level
# Quarentine
# Virus halo
# Revisar lo del árbol de búsqueda rápida y los estados de los agentes
# ... que no cambien esos estados!

class Population():
    """
    """
    def __init__(
        self,
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
                # Define label
                agent=agent_index,

                # Define positions as random
                x=(self.xmax - self.xmin) * np.random.random_sample() + self.xmin,
                y=(self.ymax - self.ymin) * np.random.random_sample() + self.ymin,

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
        keys = list(self.population[0].__dict__.keys())
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
        # Update diagnosis and hospitalization states

        # Cycle runs along all agents in current population
        for agent_index in self.population.keys():
            self.population[agent_index].update_diagnosis_state(self.step)
            self.population[agent_index].update_hospitalization_state()

        #=============================================
        # Change population states by means of state transition
        self.state_transition()

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

            self.population[agent_index].move(self.dt, self.maximum_free_random_speed, self.xmin, self.xmax, self.ymin, self.ymax)

        #=============================================
        # Populate DataFrame
        self.populate_df()


    def check_dead_agents(self):
        """
        """     
        # Copy self.population using deepcopy in order to prevent modifications on self.population
        population = copy.deepcopy(self.population)
        
        # Cycle runs along all agents in current population
        for agent_index in self.population.keys():
            
            # Check if agent is dead
            dead = population[agent_index].age_and_death_verification(self.dt_scale_in_years)
            
            if dead:
                # Remove agent and catch its value using pop()
                removed_agent = population.pop(agent_index)
                
                # Add agent to agents_info_df
                agent_dict = removed_agent.__dict__
                
                self.populate_df(agent_dict)
                
        # Copy population into self.population
        self.population = copy.deepcopy(population)


    def state_transition(self):
        """
        """
        # Cycle runs along all agents in current population
        for agent_index in self.population.keys():

            self.population[agent_index].state_time += self.dt

            if (self.population[agent_index].state_max_time
                and self.population[agent_index].state_time == self.population[agent_index].state_max_time):

                # Verify: becomes into ? ... Throw the dice
                dice = np.random.random_sample()

                cummulative_probability = 0

                for (probability, becomes_into_state) in sorted(
                    zip(
                        states_transitions_by_group_dict[
                            self.population[agent_index].group
                        ][self.population[agent_index].state]['transition_probability'],
                        states_transitions_by_group_dict[
                            self.population[agent_index].group
                        ][self.population[agent_index].state]['becomes_into']
                    ),
                    # In order to be in descending order
                    reverse=True
                ):
                    cummulative_probability += probability

                    if dice <= cummulative_probability:
                        
                        self.population[agent_index].state = becomes_into_state
                        
                        self.population[agent_index].update_diagnosis_state(self.step, changed_state=True)

                        self.population[agent_index].state_time = 0
                        
                        if states_time_functions_dict[
                            self.population[agent_index].group][becomes_into_state]['time_function']:

                            self.population[agent_index].state_max_time = states_time_functions_dict[
                                self.population[agent_index].group][becomes_into_state]['time_function']()

                        else:
                            self.population[agent_index].state_max_time = None
                        
                        break


    def spatial_tress_by_state(self):
        """
        """
        #=============================================
        # Create a dictionary for saving KDTree for agents of each state
        self.spatial_trees = {}

        # Create a dictionary for saving indices for agents of each state
        self.agents_indices_by_state = {}

        for state in states_contagion_dynamics.keys():
    
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

            self.spatial_trees[state] = spatial.KDTree(points) if len(points) is not 0 else None
            self.agents_indices_by_state[state] = agents_indices if len(agents_indices) is not 0 else None


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
            if states_contagion_dynamics[self.population[agent_index].state]['can_get_infected']:

                # Retrieve agent location
                agent_location = [self.population[agent_index].x, self.population[agent_index].y]

                # List to save who infected the agent
                infected_by = []

                # Cycle through each spreader to see if the agent gets infected by the spreader
                for state in states_contagion_dynamics.keys():

                    if states_contagion_dynamics[state]['can_spread'] and self.spatial_trees[state]:

                        # Detect if any spreader is inside a distance equal to the corresponding spread_radius
                        points_inside_radius = self.spatial_trees[state].query_ball_point(
                            agent_location,
                            states_contagion_dynamics[state]['spread_radius']
                        )

                        spreaders_indices_inside_radius = self.agents_indices_by_state[state][points_inside_radius]
                        
                        # If agent_index in spreaders_indices_inside_radius, then remove it
                        if agent_index in spreaders_indices_inside_radius:

                            spreaders_indices_inside_radius = np.setdiff1d(
                                spreaders_indices_inside_radius,
                                agent_index
                            )

                        # Calculate joint probability for contagion
                        joint_probability = (1.0 - self.population[agent_index].inmunization_level) \
                                        * contagion_susceptibility_groups_probabilities[self.population[agent_index].group] \
                                        * states_contagion_dynamics[state]['spread_probability']

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

                    cummulative_probability = 0

                    for (probability, becomes_into_state) in sorted(
                        zip(
                            contagion_states_transitions_dict[
                                self.population[agent_index].state]['transition_probability'],
                            contagion_states_transitions_dict[
                                self.population[agent_index].state]['becomes_into']
                        ),
                        # In order to be in descending order
                        reverse=True
                    ):
                        cummulative_probability += probability

                        if dice <= cummulative_probability:

                            self.population[agent_index].state = becomes_into_state

                            self.population[agent_index].update_diagnosis_state(self.step, changed_state=True)

                            self.population[agent_index].state_time = 0

                            if states_time_functions_dict[
                                self.population[agent_index].group][becomes_into_state]['time_function']:

                                self.population[agent_index].state_max_time = states_time_functions_dict[
                                    self.population[agent_index].group][becomes_into_state]['time_function']()
                            else:
                                self.population[agent_index].state_max_time = None

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
            if state_alertness_dynamics_by_group[
                self.population[agent_index].group][self.population[agent_index].state]['should_be_alert']:
                
                # Retrieve agent location
                agent_location = [self.population[agent_index].x, self.population[agent_index].y]

                # Initialize "alerted by"
                self.population[agent_index].alerted_by = []

                # Cycle through each state of the neighbors to see if the agent should be alert
                for state in states_contagion_dynamics.keys():

                    # Note that an 'avoidable_agent' depends on its state but also on each group,
                    # it means that each group defines which state is avoidable
                    if (state_avoidness_dynamics_by_group[
                        self.population[agent_index].group][state]['avoidable_agent']
                    and self.spatial_trees[state]):
                        
                        # Detect if any avoidable agent is inside a distance equal to the corresponding spread_radius
                        points_inside_radius = self.spatial_trees[state].query_ball_point(
                            agent_location,
                            state_avoidness_dynamics_by_group[
                                self.population[agent_index].group][state]['avoidness_radius']
                        )

                        avoidable_indices_inside_radius = self.agents_indices_by_state[state][points_inside_radius]
                        
                        # If agent_index in avoidable_indices_inside_radius, then remove it
                        if agent_index in avoidable_indices_inside_radius:
                            
                            avoidable_indices_inside_radius = np.setdiff1d(
                                avoidable_indices_inside_radius,
                                agent_index
                            )

                        if len(avoidable_indices_inside_radius) is not 0:
                            
                            for avoidable_agent_index in avoidable_indices_inside_radius:
                                
                                # Must agent be alert ? ... Throw the dice
                                dice = np.random.random_sample()
                                
                                # Note that alertness depends on a probability, which tries to model
                                # the probability that an agent with a defined group and state is alert
                                if (dice <= state_alertness_dynamics_by_group[
                                    self.population[agent_index].group
                                ][self.population[agent_index].state]['alertness_probability']):
                                    
                                    # Agent is alerted !!!
                                    self.population[agent_index].alertness = True
                                    
                                    # Append avoidable_agent_index in alerted_by
                                    self.population[agent_index].alerted_by.append(avoidable_agent_index)

                #=============================================                    
                # Change movement direction if agent's alertness = True
                if self.population[agent_index].alertness:
                    
                    # Retrieve velocities of the agents to be avoided
                    velocities_to_avoid = [
                        np.array([self.population[agent_to_be_avoided].vx, self.population[agent_to_be_avoided].vy])
                        for agent_to_be_avoided in self.population[agent_index].alerted_by
                    ]

                    # Change movement direction
                    self.population[agent_index].alert_avoid_agents(velocities_to_avoid)

    
    def populate_df(self, agent_dict: dict=None):
        """
        """
        if not agent_dict:
            # Cycle runs along all agents in current population
            population_array = []
            
            for agent_index in self.population.keys():
                agent_dict = self.population[agent_index].__dict__
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


    def go_agent_scatter(self, agent_dict: dict):
        """
        """
        live_state = agent_dict['live_state']
        
        agent_label = agent_dict['agent']
        x = agent_dict['x']
        y = agent_dict['y']
        vx = agent_dict['vx']
        vy = agent_dict['vy']
        group = agent_dict['group']
        state = agent_dict['state']
        diagnosed = agent_dict['diagnosed']
        infected_by = agent_dict['infected_by']
        
        if live_state == 'alive':

            template = (
                f'<b>Agent</b>: {agent_label}'
                '<br>'
                f'<b>Position</b>: ({x:.2f}, {y:.2f})'
                '<br>'
                f'<b>Velocity</b>: ({vx:.2f}, {vy:.2f})'
                '<br>'
                f'<b>Group</b>: {group}'
                '<br>'
                f'<b>State</b>: {state}'
                '<br>'
                f'<b>Diagnosed</b>: {diagnosed}'
                '<br>'
                f'<b>Infected by</b>: {infected_by}'
            )

            return go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                marker_line_width=agent_marker_line_width,
                marker_symbol=population_possible_groups_markers[group],
                marker=population_color_dicts[group][state],
                text=template,
                hoverinfo='text'
            )
        
        elif live_state == 'natural death':
            
            template = (
                f'<b>Agent</b>: {agent_label}'
                '<br>'
                f'<b>Position</b>: ({x:.2f}, {y:.2f})'
                '<br>'
                f'<b>Group</b>: {group}'
                '<br>'
                f'<b>State</b>: {state}'
                '<br>'
                f'<b>Live state</b>: {live_state}'
            )

            return go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                marker_line_width=agent_marker_line_width,
                marker_symbol=natural_death,
                marker=dead_color_dict,
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
                f'<b>Group</b>: {group}'
                '<br>'
                f'<b>State</b>: {state}'
                '<br>'
                f'<b>Live state</b>: {live_state}'
            )

            return go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                marker_line_width=agent_marker_line_width,
                marker_symbol=dead_by_disease,
                marker=dead_color_dict,
                text=template,
                hoverinfo='text'
            )
    

    def plot_current_locations(self):
        """
        """
        fig = go.Figure(
            layout=go.Layout(
                xaxis=dict(range=[self.xmin, self.xmax], autorange=False, zeroline=False),
                yaxis=dict(range=[self.ymin, self.ymax], autorange=False, zeroline=False),
                title_text='Current Population Locations',
                hovermode='closest'
            )
        )
        
        population_data = self.agents_info_df.loc[
            self.agents_info_df['step'] == self.step
        ].sort_values(by=['agent']).to_dict(orient='records')
        
        # Cycle runs along all agents in current population
        for agent_dict in population_data:           
            # Add traces
            fig.add_trace(
                self.go_agent_scatter(agent_dict)
            )
        
        fig.update_layout(showlegend=False)

        fig.show()
        

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
            
            full_data.append([self.go_agent_scatter(population_data[i]) for i in range(len(population_data))])
        
        # Max length
        max_length = (self.xmax - self.xmin) if (self.xmax - self.xmin) > (self.ymax - self.ymin) else (self.ymax - self.ymin)
        
        # Create figure
        fig = go.Figure(
            
            data=full_data[0],
            
            layout=go.Layout(
                width=600 * (self.xmax - self.xmin)/max_length, 
                height=600 * (self.ymax - self.ymin)/max_length,
                xaxis=dict(range=[self.xmin, self.xmax], autorange=False, zeroline=False),
                yaxis=dict(range=[self.ymin, self.ymax], autorange=False, zeroline=False),
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

def initial_population_group_state_scene(
    population_number,
    vulnerable_number
):
    population_group_list = [
        population_possible_groups[1] if i < vulnerable_number else population_possible_groups[0]
        for i in range(population_number)
    ]
    population_state_list = [
        population_possible_states[1] if i%3 == 0 else population_possible_states[0]
        for i in range(population_number)
    ]
    
    return population_group_list, population_state_list

def go_line(x: list, y: list, name: str):
    return go.Scatter(
        x=x,
        y=y,
        mode='lines',
        line=line_color_dicts['vulnerable'][name],
        name=name
    )

def agents_times_series_plot(
    agents_dataframe,
    mode: str='states',
    column: str=None,
    inspection_values: list=None
):
    """
        mode: str
            
    """
    if mode == 'states':

        df = agents_dataframe.loc[
               agents_dataframe['live_state'] == 'alive' 
            ][['step', 'agent', 'state']].groupby(['step', 'state'], as_index=False).count().copy()

        df.rename(columns={'agent':'agents'}, inplace=True)
        
        states = agents_dataframe['state'].unique()
        
        fig = go.Figure()
        
        for state in states:
            subdf = df.loc[
                df['state'] == state
            ][['step', 'agents']].copy()

            # Add traces
            fig.add_trace(
                go_line(
                    x=subdf['step'].to_list(),
                    y=subdf['agents'].to_list(),
                    name=state
                )
            )

        fig.update_xaxes(rangeslider_visible=True)
        
        fig.update_layout(hovermode='x unified')

        fig.show()


if __name__ == "__main__":

    population_group_list, population_state_list = initial_population_group_state_scene(
        population_number,
        vulnerable_number
    )

    agents = Population(
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
    agents.plot_current_locations()

    start_time = time.time()

    for i in range(100):
        agents.evolve_population()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    agents.plot_current_locations()

    agents.animate_population()

    agents_times_series_plot(agents.agents_info_df)




