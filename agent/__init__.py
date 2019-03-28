from agent.agents import Agent2x, Agent3x


def get_training_agent(config, net):
    assert config.name is not None
    if config.name == 'skeleton' or config.name == 'view':
        return Agent2x(config, net)
    else:
        return Agent3x(config, net)
