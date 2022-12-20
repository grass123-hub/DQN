"""
INFER DQN ONNX.
"""
import argparse
import gym
import onnxruntime
import numpy as np

parser = argparse.ArgumentParser(description='MindSpore Reinforcement DQN Onnx Infer')
parser.add_argument('--onnx_file', type=str, default='./scripts/onnx/policy_net/policy_net_1000.onnx',
                    help='onnx file location.')
options, _ = parser.parse_known_args()


def run_eval():
    '''DQN onnx infer entry'''
    onnx_file = options.onnx_file
    dqn_session = onnxruntime.InferenceSession(onnx_file, provider_options='CUDAExecutionProvider')
    env = gym.make('CartPole-v0')
    env.seed(1)
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        state = np.expand_dims(state, 0)
        inputs = {dqn_session.get_inputs()[0].name: state}
        outs = dqn_session.run(None, inputs)
        action = outs[0].item()
        state, r, done, _ = env.step(action)
        total_reward += r
    print('-----------------------------------------')
    print('Evaluate result is {}, onnx file in {}'.format(total_reward, onnx_file))
    print('-----------------------------------------')
    print('eval end')
    env.close()


if __name__ == '__main__':
    run_eval()
