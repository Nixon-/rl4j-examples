package org.deeplearning4j.rl4j;

import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CConfiguration;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscreteDense;
import org.deeplearning4j.rl4j.mdp.gym.GymEnv;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactorySeparateStdDense;
import org.deeplearning4j.rl4j.space.Box;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/18/16.
 *
 * main example for A3C on cartpole
 *
 */
public class A3CCartpole {

    private static A3CConfiguration CARTPOLE_A3C =
            new A3CConfiguration(
                    123,            //Random seed
                    200,            //Max step By epoch
                    500000,         //Max step
                    16,              //Number of threads
                    5,              //t_max
                    10,             //num step noop warmup
                    0.01,           //reward scaling
                    0.99,           //gamma
                    10.0           //td-error clipping
            );



    private static final ActorCriticFactorySeparateStdDense.Configuration CARTPOLE_NET_A3C =
            new ActorCriticFactorySeparateStdDense.Configuration(
            3,                      //number of layers
            16,                     //number of hidden nodes
            0.001,                 //learning rate
            0.000                   //l2 regularization
    );


    public static void main( String[] args )
    {
        A3CcartPole();
    }

    private static void A3CcartPole() {

        //record the training data in rl4j-data in a new folder
        DataManager manager = new DataManager(true);

        //define the mdp from gym (name, render)
        GymEnv<Box, Integer, DiscreteSpace> mdp = new GymEnv<>("CartPole-v0", false, false);

        //define the training
        A3CDiscreteDense<Box> dql = new A3CDiscreteDense<>(mdp, CARTPOLE_NET_A3C, CARTPOLE_A3C, manager);

        //start the training
        dql.train();

        //close the mdp (http connection)
        mdp.close();

    }



}
