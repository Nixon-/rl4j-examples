package org.deeplearning4j.rl4j;


import org.deeplearning4j.rl4j.learning.sync.qlearning.QLConfiguration;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.mdp.gym.GymEnv;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.space.Box;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.util.DataManager;

import java.util.logging.Logger;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/11/16.
 *
 * Main example for Cartpole DQN
 *
 * **/
public class Cartpole
{
    private final static QLConfiguration CARTPOLE_QL =
            new QLConfiguration(
                    123,    //Random seed
                    200,    //Max step By epoch
                    150000, //Max step
                    150000, //Max size of experience replay
                    32,     //size of batches
                    500,    //target update (hard)
                    10,     //num step noop warmup
                    0.01,   //reward scaling
                    0.99,   //gamma
                    1.0,    //td-error clipping
                    0.1f,   //min epsilon
                    1000,   //num step for eps greedy anneal
                    true    //double DQN
            );

    private final static DQNFactoryStdDense.Configuration CARTPOLE_NET =
            new DQNFactoryStdDense.Configuration(
                    3,         //number of layers
                    16,        //number of hidden nodes
                    0.001,     //learning rate
                    0.00       //l2 regularization
            );

    public static void main( String[] args )
    {
        cartPole();
        loadCartpole();
    }

    private static void cartPole() {

        //record the training data in rl4j-data in a new folder (save)
        DataManager manager = new DataManager(true);

        //define the mdp from gym (name, render)
        GymEnv<Box, Integer, DiscreteSpace> mdp = new GymEnv<>("CartPole-v0", false, false);

        //define the training
        QLearningDiscreteDense<Box> dql = new QLearningDiscreteDense<>(mdp, CARTPOLE_NET, CARTPOLE_QL, manager);

        //train
        dql.train();

        //get the final policy
        DQNPolicy<Box> pol = dql.getPolicy();

        //serialize and save (serialization showcase, but not required)
        pol.save("/tmp/pol1");

        //close the mdp (close http)
        mdp.close();
    }

    private static void loadCartpole(){

        //showcase serialization by using the trained agent on a new similar mdp (but render it this time)

        //define the mdp from gym (name, render)
        GymEnv<Box, Integer, DiscreteSpace> mdp2 = new GymEnv<>("CartPole-v0", true, false);

        //load the previous agent
        DQNPolicy<Box> pol2 = DQNPolicy.load("/tmp/pol1");

        //evaluate the agent
        double rewards = 0;
        for (int i = 0; i < 1000; i++) {
            mdp2.reset();
            double reward = pol2.play(mdp2);
            rewards += reward;
            Logger.getAnonymousLogger().info("Reward: " + reward);
        }

        Logger.getAnonymousLogger().info("average: " + rewards/1000);
    }
}
