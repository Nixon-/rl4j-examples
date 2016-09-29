package org.deeplearning4j.rl4j;

import org.deeplearning4j.rl4j.learning.async.nstep.discrete.AsyncNStepQLearningDiscrete;
import org.deeplearning4j.rl4j.learning.async.nstep.discrete.AsyncNStepQLearningDiscreteDense;
import org.deeplearning4j.rl4j.mdp.gym.GymEnv;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.space.Box;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/18/16.
 *
 * main example for Async NStep QLearning on cartpole
 */
public class AsyncNStepCartpole {


    public static AsyncNStepQLearningDiscrete.AsyncNStepQLConfiguration CARTPOLE_NSTEP =
            new AsyncNStepQLearningDiscrete.AsyncNStepQLConfiguration(
                    123,     //Random seed
                    200,     //Max step By epoch
                    300000,  //Max step
                    16,      //Number of threads
                    5,       //t_max
                    100,     //target update (hard)
                    10,      //num step noop warmup
                    0.01,    //reward scaling
                    0.99,    //gamma
                    100.0,   //td-error clipping
                    0.1f,    //min epsilon
                    9000     //num step for eps greedy anneal
            );

    public static DQNFactoryStdDense.Configuration CARTPOLE_NET_NSTEP =
            new DQNFactoryStdDense.Configuration(
                    3,         //number of layers
                    16,        //number of hidden nodes
                    0.0005,    //learning rate
                    0.001      //l2 regularization
            );


    public static void main( String[] args )
    {
        cartPole();
    }


    public static void cartPole() {

        //record the training data in rl4j-data in a new folder
        DataManager manager = new DataManager(true);

        GymEnv<Box, Integer, DiscreteSpace> mdp = new GymEnv<>("CartPole-v0", false, false);

        //define the training
        AsyncNStepQLearningDiscreteDense<Box> dql =
                new AsyncNStepQLearningDiscreteDense<>(mdp, CARTPOLE_NET_NSTEP, CARTPOLE_NSTEP, manager);

        //train
        dql.train();

        //close the mdp (close connection)
        // client.close();
    }


}
