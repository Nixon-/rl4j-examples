package org.deeplearning4j.rl4j;


import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.async.AsyncLearning;
import org.deeplearning4j.rl4j.learning.async.nstep.discrete.NStepQLearningDiscreteDense;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.mdp.toy.HardDeteministicToy;
import org.deeplearning4j.rl4j.mdp.toy.SimpleToy;
import org.deeplearning4j.rl4j.mdp.toy.SimpleToyState;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.util.Constants;
import org.deeplearning4j.rl4j.util.DataManager;

import static org.deeplearning4j.rl4j.AsyncNStepCartpole.CARTPOLE_A3C;
import static org.deeplearning4j.rl4j.Cartpole.CARTPOLE_NET;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/11/16.
 */
public class Toy {


    public static QLearning.QLConfiguration TOY_QL =
            new QLearning.QLConfiguration(
                    123, //seed
                    100000, //maxEpochStep
                    80000, //maxStep
                    10000, //expRepMaxSize
                    32, //batchSize
                    100, //targetDqnUpdateFreq
                    0, //updateStart
                    0.99, //gamma
                    10.0, //errorClamp
                    0.1f, //minEpsilon
                    1f / 2000f, //epsilonDecreaseRate
                    true //doubleDQN
            );

    public static AsyncLearning.AsyncConfiguration TOY_ASYNC_QL =
            new AsyncLearning.AsyncConfiguration(
                    123, //seed
                    100000, //maxEpochStep
                    80000, //maxStep
                    8,
                    10, //batchSize
                    0.99, //gamma
                    100,
                    100, //errorClamp
                    10f,
                    0.1f, //minEpsilon
                    1f / 2000f //epsilonDecreaseRate
            );

    public static DQNFactoryStdDense.Configuration TOY_NET =
            new DQNFactoryStdDense.Configuration(4, 15, 0.01, 0.00, 0.99);

    public static void main( String[] args )
    {
        //simpleToy();
        toyAsync();

    }

    public static void simpleToy() {
        DataManager manager = new DataManager();
        SimpleToy mdp = new SimpleToy(20);
        Learning<SimpleToyState, Integer, DiscreteSpace, IDQN> dql = new QLearningDiscreteDense<SimpleToyState>(mdp, TOY_NET, TOY_QL, manager);
        mdp.setFetchable(dql);
        dql.train();
        dql.getPolicy();
        mdp.close();
    }

    public static void hardToy() {
        DataManager manager = new DataManager();
        MDP mdp = new HardDeteministicToy();
        ILearning<SimpleToyState, Integer, DiscreteSpace> dql = new QLearningDiscreteDense(mdp, TOY_NET, TOY_QL, manager);
        dql.train();
        dql.getPolicy();
        mdp.close();
    }


    public static void toyAsync() {
        DataManager manager = new DataManager();
        //GymEnv mdp = new GymEnv("CartPole-v0",  false);
        SimpleToy mdp = new SimpleToy(20);
        NStepQLearningDiscreteDense dql = new NStepQLearningDiscreteDense<SimpleToyState>(mdp, TOY_NET, TOY_ASYNC_QL, manager);
        mdp.setFetchable(dql);
        dql.train();
        dql.getPolicy();
        mdp.close();
    }

}
