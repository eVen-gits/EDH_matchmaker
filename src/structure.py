from __future__ import annotations


from functools import partial

class TournamentStructure:
    DEFAULT_STATES = {
        'Setup':{
            'label': 'Setup',
            'actions': {
                'select_logic': {
                    'label': 'Select logic',
                    'dest': 'TournamentReady'
                }
            }
        },
        'TournamentReady': {
            'label': 'Tournament ready',
            'actions': {
                'add_player': {
                    'label': 'Add player',
                    'dest': 'TournamentReady',
                },
                'remove_player': {
                    'label': 'Remove player',
                    'dest': 'TournamentReady',
                },
                'bench_player': {
                    'label': 'Bench player',
                    'dest': 'TournamentReady',
                },
                'drop_player': {
                    'label': 'Drop player',
                    'dest': 'TournamentReady'
                },
                'create_pods': {
                    'label': 'Create pods',
                    'dest': 'PodsCreated'
                }

            }
        },
        'PodsCreated': {
            'label': 'Pods created',
            'actions': {
                'move_player': {
                    'label': 'Move player',
                    'dest': 'PodsCreated',
                },
                'drop_player': {
                    'label': 'Drop player',
                    'dest': 'PodsCreated',
                },
                'bench_player': {
                    'label': 'Bench player',
                    'dest': 'PodsCreated',
                },
                'reset_pods': {
                    'label': 'Reset pods',
                    'dest': 'TournamentReady',
                },
                'start_round': {
                    'label': 'Start round',
                    'dest': 'RoundStarted',
                },
            }
        },
        'RoundStarted': {
            'label': 'Round started',
            'actions': {
                'assign_game_loss': {
                    'label': 'Assign game loss',
                    'dest': 'RoundStarted'},
                'report_win': {
                    'label': 'Report win',
                    'dest': 'RoundStarted'},
                'report_loss': {
                    'label': 'Report loss',
                    'dest': 'RoundStarted'},
                'report_draw': {
                    'label': 'Report draw',
                    'dest': 'RoundStarted'},
                'end_round': {
                    'label': 'End round',
                    'dest': 'TournamentReady'
                },
            }
        }
    }

    def __init__(self, states=DEFAULT_STATES):
        self.states = {}
        self.state: TState = TState('invalid', 'Placeholder state')
        self.init('Setup', states)

    def init(self, start_state: str, states_dict: dict):
        for key, value in states_dict.items():
            state = self.add_state(TState(key, value['label']))

        for state_ref, state in states_dict.items():
            for act_ref, data in state['actions'].items():
                self.add_action(act_ref, data['label'], state_ref, data['dest'])
        self.state = self.states[start_state]

    def add_state(self, state: TState):
        self.states[state.ref] = state
        return state

    def add_action(self, ref: str, name: str, start_ref: str, end_ref: str):
        start = self.states[start_ref]
        end = self.states[end_ref]
        action = TAction(ref, name, start, end)
        setattr(self, ref, partial(action, self))  # Dynamically add action method
        return action

    @property
    def actions(self):
        return self.state.actions

    def __getattr__(self, name):
        # Handle dynamic action attributes
        if name in self.state._actions:
            return partial(self._call_action, self.state._actions[name])
        return object.__getattribute__(self, name)

    def __repr__(self, var='name'):
        return '\n'.join([v.__repr__(var) for k, v in self.states.items()])


class TState:
    def __init__(self, ref: str, name: str):
        self.ref = ref
        self.name = name
        self._actions = {}

    @property
    def actions(self):
        return list(self._actions.keys())

    def add_action(self, action: TAction):
        self._actions[action.ref] = action

    def __getattr__(self, name):
        # Handle dynamic action attributes
        if name in self._actions:
            return self._actions[name]
        return object.__getattribute__(self, name)

    def __repr__(self, var='name'):
        return "{}\n\t{}".format(
            self.name,
            '\n\t'.join(
                [f"{getattr(v, var)} -> {getattr(v.end, var)}" for k,
                 v in self._actions.items()]
            )
        )


class TAction:
    def __init__(self, ref: str, name: str, start: TState, end: TState):
        self.ref = ref
        self.name = name
        self.start = start
        self.end = end

        start.add_action(self)

    def __call__(self, caller: TournamentStructure):
        print(f"{self.start.name} -- {self.name} --> {self.end.name}")
        caller.state = self.end
        #print(caller.__repr__(var='ref'))

# Example usage

if __name__ == '__main__':
    t = TournamentStructure()
    print(t)
    print(t.state.__repr__(var='ref'))
    # t.exec('create_pods')
    t.create_pods()  # No need to pass `t` explicitly
    print(t.state.__repr__(var='ref'))