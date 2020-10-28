# CDS Library

Contagious Disease Simulation - Library

## Install

Install with the command

    pip install -e .

----

## TODO

- `plot_locations()`
  - Step or datetime info in title?

- `animate_population()`
  - Save in other video formats
  - Step or datetime info in title?
  - Change velocity, frames, menu, axis labels

- `go_agent_scatter()`
  - Add **quarantine group** in point label
  - Add **quarantine state** in point label

- In `basic_population.py`
  - Parallelization
  - Add units and time units
  - Improve diagnosis state?? (using networks)
  - Initial spatial distribution
  - Family groups and networks
  - Virus halo ... Probability density
  - Change probability of transition state by exposition to virus

- `update_diagnosis_state()`
  - Improve the case when `self.__quarantine_after_being_diagnosed_enabled == False`
  to model the time that has to pass before to be diagnosed again ... Right now
  it is an infinite time.

- `update_hospitalization_state()`
  - When an agent is removed from hospitalization, change the positions assigned
  in order to use random if hospital location is None

- Fix `population_df_path` ... It seems that the problem was due to visual code
