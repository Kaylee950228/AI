package problem;

import simulator.*;

import java.io.IOException;
import java.util.*;


public class MCTS {

    private ProblemSpec ps;
    private Helper helper;
    private Map<Terrain, Object[]> terrain2spec;

    public MCTS(ProblemSpec ps) {
        this.ps = ps;
        this.helper = new Helper();
        this.terrain2spec = helper.getBestSpecification();
    }

    // Define 3 grame state: win, lose, running
    public enum GameState {
        WIN,
        LOSE,
        RUNNING
    }

    // Define the state node data structure of the Monte Carlo Search Tree,

    // If all the action are deterministic, we would use a HashMap<Action, StateNode> to store all the actions and
    //   their corresponding next state. However, the action MOVE is indeterministic (may move -4..5 cells), which
    //   means that the action MOVE has more than one possible next states. It contradicts with HashMap since the
    //   key-value pair is one-to-one. (In this case, key(MOVE) corresponds to at most 12 values (possible next states).

    // Therefore, We defined two kinds of nodes: ActionNode and StateNode.
    // They appear in the Monte Carlo Search Tree in an interleaving manner. That is, the first level are all state nodes,
    // then the second level are all action nodes, the third level are all state nodes...
    //
    // Each state node maintains a HashMap<Action, ActionNode> to store all the possible action and its corresponding action nodes.
    // In this case, action MOVE only corresponds to a single ActionNode, all the possible outcomes are stored as the children of
    //   the state node.
    //
    // Each action node maintains a HashMap<StateTuple, StateNode> to store all the possible state tuples and its corresponding state nodes.

    class StateNode {
        private StateTuple stateTuple; // stores state and whether the state are recovered from Slip or Breakdown
        private HashMap<Action, ActionNode> map; // store all the possible actions and their corresponding action nodes
        private ActionNode parent; // which action lead to current state

        // What is current timestep, it will be used in the Simulation (roll-out) phase of MCTS.
        // Suppose MAXT=30 and current timestep is 20.
        // During Simulation, if it doesn't win in (30-20=10) steps, the simulation fails.
        // If we don't store current timestep, we don't how many steps are available in the Simulation phase.
        private int timestep;

        private int visit; // how many time this node has been visited
        private int reward; // what is the reward of this node

        public int getVisit() {
            return visit;
        }

        public StateTuple getStateTuple() {
            return stateTuple;
        }

        public HashMap<Action, ActionNode> getMap() {
            return map;
        }

        public ActionNode getParent() {
            return parent;
        }

        public int getTimestep() {
            return timestep;
        }

        public StateNode(StateTuple stateTuple, ActionNode parent, int timestep) {
            this.stateTuple = stateTuple;
            this.map = new HashMap<>();
            this.parent = parent;
            this.visit = 0;
            this.reward = 0;
            this.timestep = timestep;
        }

        public double getRatio() {
            return (double) reward / visit;
        }

        public void increaseVisit() {
            this.visit++;
        }

        public void increaseReward(double result) {
            this.reward += result;
        }

        public GameState getGameState() {
            // Reached the goal
            if (this.stateTuple.state.getPos() >= ps.getN()) {
                return GameState.WIN;
            } else if (this.timestep >= ps.getMaxT()) {
                // Time out
                return GameState.LOSE;
            } else {
                // Otherwise the game is still running
                return GameState.RUNNING;
            }
        }

        public int getReward() {
            return reward;
        }

    }

    class ActionNode {
        private Action action;
        private HashMap<StateTuple, StateNode> map;
        private StateNode parent;
        private double reward;
        private int visit;

        public Action getAction() {
            return this.action;
        }

        public HashMap<StateTuple, StateNode> getMap() {
            return map;
        }

        public StateNode getParent() {
            return parent;
        }

        public double getReward() {
            return reward;
        }

        public int getVisit() {
            return visit;
        }

        public ActionNode(Action action, StateNode parent) {
            this.action = action;
            this.map = new HashMap<>();
            this.parent = parent;
            this.visit = 0;
        }

        public double getRatio() {
            return reward / this.visit;
        }

        public void increaseReward(double result) {
            this.reward += result;
        }

        public void increaseVisit() {
            this.visit++;
        }

        @Override
        public String toString() {
            return "ActionNode{" +
                    "action=" + action +
                    ", visit=" + visit +
                    ", ratio=" + getRatio() +
                    '}';
        }
    }

    class Helper {
        // Given current state, return the fuel consumption by action MOVE,
        // copied from support code
        public int getFuelConsumption(State currentState) {
            // get parameters of current state
            Terrain terrain = ps.getEnvironmentMap()[currentState.getPos() - 1];
            String car = currentState.getCarType();
            TirePressure pressure = currentState.getTirePressure();

            // get fuel consumption
            int terrainIndex = ps.getTerrainIndex(terrain);
            int carIndex = ps.getCarIndex(car);
            int fuelConsumption = ps.getFuelUsage()[terrainIndex][carIndex];

            if (pressure == TirePressure.FIFTY_PERCENT) {
                fuelConsumption *= 3;
            } else if (pressure == TirePressure.SEVENTY_FIVE_PERCENT) {
                fuelConsumption *= 2;
            }
            return fuelConsumption;
        }

        // Given problemSpec and current state, return all possible actions in a set.
        // For example, if the fuel is less than fuel consumption, action MOVE
        //   won't appear in possible actions.

        // Problem: maybe too slow?
        //
        //        : adding A7 and A8, the number of possible actions increased significantly,
        //            the probability of choosing the action MOVE is quite low.
        //
        //          For example, suppose there are 25 possible actions, and MaxT=100.
        //
        //          The (maximum) expectation of the number of MOVE chosen is 4.
        //            (if slip or breakdown occurs, we don't have 100 times to perform the action)
        //
        //          If N>20, it is unlikely to reach the goal even every time we MOVE 5 cells.

        public Set<Action> getAllPossibleActions(ProblemSpec ps, StateTuple stateTuple) {
            State currentState = stateTuple.state;
            Set<Action> actions = new HashSet<>();

            // A1. Continue Moving
            if (currentState.getFuel() > getFuelConsumption(currentState)) {
                if (currentState.getFuel() >= getFuelConsumption(currentState)) {
                    actions.add(new Action(ActionType.MOVE));
                }
            }

            if (ps.getLevel().getLevelNumber() <= 3) {
                // A2. Change the car type
                for (String carType : ps.getCarOrder()) {
                    if (!carType.equals(currentState.getCarType())) {
                        actions.add(new Action(ActionType.CHANGE_CAR, carType));
                    }
                }
                // A3. Change the driver
                for (String driver : ps.getDriverOrder()) {
                    if (!driver.equals(currentState.getDriver())) {
                        actions.add(new Action(ActionType.CHANGE_DRIVER, driver));
                    }
                }
                // A4. Change the tire(s) of existing car
                for (Tire tire : ps.getTireOrder()) {
                    if (!tire.equals(currentState.getTireModel())) {
                        actions.add(new Action(ActionType.CHANGE_TIRES, tire));
                    }
                }

                // Only support level 1
                if (ps.getLevel().getLevelNumber() == 1) {
                    return actions;
                }

                // A5. Add fuel to existing car
                actions.add(new Action(ActionType.ADD_FUEL, 10));

                // A6. Change pressure to the tires
                actions.add(new Action(ActionType.CHANGE_PRESSURE, TirePressure.FIFTY_PERCENT));
                actions.add(new Action(ActionType.CHANGE_PRESSURE, TirePressure.SEVENTY_FIVE_PERCENT));
                actions.add(new Action(ActionType.CHANGE_PRESSURE, TirePressure.ONE_HUNDRED_PERCENT));
                actions.remove(new Action(ActionType.CHANGE_PRESSURE, currentState.getTirePressure()));

//                 Only support level 2 & 3

            } else if (ps.getLevel().getLevelNumber() == 4) {
                // level 4
                // A7. Combination of options A2 and A3.
                for (String carType : ps.getCarOrder()) {
                    for (String driverType : ps.getDriverOrder()) {
                        actions.add(new Action(ActionType.CHANGE_CAR_AND_DRIVER, carType, driverType));
                    }
                }
                actions.remove(new Action(ActionType.CHANGE_CAR_AND_DRIVER, currentState.getCarType(), currentState.getDriver()));

                // A4. Change the tire(s) of existing car
                for (Tire tire : ps.getTireOrder()) {
                    if (!tire.equals(currentState.getTireModel())) {
                        actions.add(new Action(ActionType.CHANGE_TIRES, tire));
                    }
                }

                // A5. Add fuel to existing car
                actions.add(new Action(ActionType.ADD_FUEL, 10));

                // A6. Change pressure to the tires
                actions.add(new Action(ActionType.CHANGE_PRESSURE, TirePressure.FIFTY_PERCENT));
                actions.add(new Action(ActionType.CHANGE_PRESSURE, TirePressure.SEVENTY_FIVE_PERCENT));
                actions.add(new Action(ActionType.CHANGE_PRESSURE, TirePressure.ONE_HUNDRED_PERCENT));
                actions.remove(new Action(ActionType.CHANGE_PRESSURE, currentState.getTirePressure()));

            } else {
                // level 5
                // A7. Combination of options A2 and A3.
                for (String carType : ps.getCarOrder()) {
                    for (String driverType : ps.getDriverOrder()) {
                        actions.add(new Action(ActionType.CHANGE_CAR_AND_DRIVER, carType, driverType));
                    }
                }
                actions.remove(new Action(ActionType.CHANGE_CAR_AND_DRIVER, currentState.getCarType(), currentState.getDriver()));

                for (Tire tire : ps.getTireOrder()) {
                    actions.add(new Action(ActionType.CHANGE_TIRE_FUEL_PRESSURE, tire, 0, TirePressure.FIFTY_PERCENT));
                    actions.add(new Action(ActionType.CHANGE_TIRE_FUEL_PRESSURE, tire, 0, TirePressure.SEVENTY_FIVE_PERCENT));
                    actions.add(new Action(ActionType.CHANGE_TIRE_FUEL_PRESSURE, tire, 0, TirePressure.ONE_HUNDRED_PERCENT));
                    actions.add(new Action(ActionType.CHANGE_TIRE_FUEL_PRESSURE, tire, 10, TirePressure.FIFTY_PERCENT));
                    actions.add(new Action(ActionType.CHANGE_TIRE_FUEL_PRESSURE, tire, 10, TirePressure.SEVENTY_FIVE_PERCENT));
                    actions.add(new Action(ActionType.CHANGE_TIRE_FUEL_PRESSURE, tire, 10, TirePressure.ONE_HUNDRED_PERCENT));
                }
                actions.remove(new Action(ActionType.CHANGE_TIRE_FUEL_PRESSURE, currentState.getTireModel(), 0, currentState.getTirePressure()));
            }

            return actions;
        }

        public int getNumPossibleActions(StateTuple stateTuple) {
            int rt;

            int numCar = ps.getCT();
            int numDriver = ps.getDT();

            if (ps.getLevel().getLevelNumber() == 1) {
                rt = 1 + numCar + numDriver + 4;
                rt -= 3;
            } else if (ps.getLevel().getLevelNumber() == 2 || ps.getLevel().getLevelNumber() == 3) {
                rt = 1 + numCar + numDriver + 4 + 1 + 3;
                rt -= 4;
            } else if (ps.getLevel().getLevelNumber() == 4) {
                rt = 1 + 4 + 1 + 3 + numCar * numDriver;
                rt -= 3;
            } else {
                rt = 1 + numCar * numDriver + 4 * 3;
                rt -= 2;
            }

            int consumption = getFuelConsumption(stateTuple.state);
            if (stateTuple.state.getFuel() < consumption) {
                rt--;
            }

            return rt;
        }

        // Given the ProblemSpec and current stateTuple, first generate all the possible actions set,
        //   then randomly pick one action (random policy in roll-out phase).

        // Might have some problem:
        // level 1: 1+2+2+4=9 N<=10 MaxT=30
        // level 2: 1+3+2+4+1+3=14 N<=10 MaxT=30
        // level 3: 1+3+2+4+1+3=14 N<=30 MaxT=90
        // level 4: 1+0+0+4+1+3+25=34 N<=30 MaxT=90
        // level 5: 1+0+0+0+0+0+25+12=38 N<=30 MaxT=90
        public Action selectRandomAction(ProblemSpec ps, StateTuple stateTuple) {
            Set<Action> actions = getAllPossibleActions(ps, stateTuple);
            Random random = new Random();
            int index = random.nextInt(actions.size());
            Iterator<Action> iter = actions.iterator();
            for (int i = 0; i < index; i++) {
                iter.next();
            }
            return iter.next();
        }

        private Action balancedSelectRandomAction(ProblemSpec ps, StateTuple stateTuple) {

            int fuel;
            String car, driver;
            Tire tire;
            TirePressure pressure;

            List<ActionType> validActionTypes = ps.getLevel().getAvailableActions();
            List<TirePressure> validPressures = new LinkedList<>();
            validPressures.add(TirePressure.FIFTY_PERCENT);
            validPressures.add(TirePressure.SEVENTY_FIVE_PERCENT);
            validPressures.add(TirePressure.ONE_HUNDRED_PERCENT);

            int numActions = validActionTypes.size();
            int CT = ps.getCT();
            int DT = ps.getDT();
            int TiT = ProblemSpec.NUM_TYRE_MODELS;
            int PressureT = ProblemSpec.TIRE_PRESSURE_LEVELS;

            ActionType actionType = validActionTypes.get(aRandomInt(0, numActions));
            Action action = null;

            boolean success = false;

            while (!success) {
                State currentState = stateTuple.state;
                switch (actionType.getActionNo()) {
                    case 1:
                        int currentFuel = currentState.getFuel();
                        if (currentFuel > getFuelConsumption(currentState)) {
                            action = new Action(actionType);
                            success = true;
                        } else {
                            actionType = validActionTypes.get(aRandomInt(0, numActions));
                        }
                        break;
                    case 2:
                        car = ps.getCarOrder().get(aRandomInt(0, CT));
                        while (car.equals(currentState.getCarType())) {
                            car = ps.getCarOrder().get(aRandomInt(0, CT));
                        }
                        action = new Action(actionType, car);
                        success = true;
                        break;
                    case 3:
                        driver = ps.getDriverOrder().get(aRandomInt(0, DT));
                        while (driver.equals(currentState.getCarType())) {
                            driver = ps.getDriverOrder().get(aRandomInt(0, DT));
                        }
                        action = new Action(actionType, driver);
                        success = true;
                        break;
                    case 4:
                        tire = ps.getTireOrder().get(aRandomInt(0, TiT));
                        while (tire.equals(currentState.getTireModel())) {
                            tire = ps.getTireOrder().get(aRandomInt(0, TiT));
                        }
                        action = new Action(actionType, tire);
                        success = true;
                        break;
                    case 5:
                        fuel = 10;
                        action = new Action(actionType, fuel);
                        success = true;
                        break;
                    case 6:
                        pressure = validPressures.get(aRandomInt(0, PressureT));
                        while (pressure.equals(currentState.getTirePressure())) {
                            pressure = validPressures.get(aRandomInt(0, PressureT));
                        }
                        action = new Action(actionType, pressure);
                        success = true;
                        break;
                    case 7:
                        car = ps.getCarOrder().get(aRandomInt(0, CT));
                        driver = ps.getDriverOrder().get(aRandomInt(0, DT));
                        action = new Action(actionType, car, driver);
                        break;
                    default:
                        tire = ps.getTireOrder().get(aRandomInt(0, TiT));
                        fuel = aRandomInt(ProblemSpec.FUEL_MIN, ProblemSpec.FUEL_MAX);
                        pressure = validPressures.get(aRandomInt(0, PressureT));
                        action = new Action(actionType, tire, fuel, pressure);
                }
            }

            return action;
        }

        private int aRandomInt(int min, int max) {

            if (min >= max) {
                throw new IllegalArgumentException("max must be greater than min");
            }

            Random r = new Random();
            return r.nextInt((max - min)) + min;
        }


        // instead of using 0/1 reward, I customized the reward function
        // If reached the goal, the reward is 1, multiplied by MaxT/timesteps (the less timesteps, the larger reward)
        // If didn't reach the goal but moved forward, scale the reward to 0-1 proportional to the moved distance
        //   (the more distance moved forward, the larger reward)
        // If moved backward, the reward is 0
        public double calculateReward(int start, int end, int timesteps) {
            double reward;
            // WIN
            if (end == ps.getN()) {
                reward = 1.0 * ((double) ps.getMaxT() / timesteps);
            } else if (end > start) {
                // Failed but move forward
                reward = (double) (end - start) / (ps.getN() - start);
            } else {
                reward = 0;
            }
            return reward;
        }

        // Copied from support code
        private double[] getMoveProbs(Terrain terrain, String car, String driver, Tire tire, TirePressure pressure) {

            // get parameters of current state
            int terrainIndex = ps.getTerrainIndex(terrain);

            // calculate priors
            double priorK = 1.0 / ProblemSpec.CAR_MOVE_RANGE;
            double priorCar = 1.0 / ps.getCT();
            double priorDriver = 1.0 / ps.getDT();
            double priorTire = 1.0 / ProblemSpec.NUM_TYRE_MODELS;
            double priorTerrain = 1.0 / ps.getNT();
            double priorPressure = 1.0 / ProblemSpec.TIRE_PRESSURE_LEVELS;

            // get probabilities of k given parameter
            double[] pKGivenCar = ps.getCarMoveProbability().get(car);
            double[] pKGivenDriver = ps.getDriverMoveProbability().get(driver);
            double[] pKGivenTire = ps.getTireModelMoveProbability().get(tire);
            double pSlipGivenTerrain = ps.getSlipProbability()[terrainIndex];
            double[] pKGivenPressureTerrain = convertSlipProbs(pSlipGivenTerrain, pressure);

            // use bayes rule to get probability of parameter given k
            double[] pCarGivenK = bayesRule(pKGivenCar, priorCar, priorK);
            double[] pDriverGivenK = bayesRule(pKGivenDriver, priorDriver, priorK);
            double[] pTireGivenK = bayesRule(pKGivenTire, priorTire, priorK);
            double[] pPressureTerrainGivenK = bayesRule(pKGivenPressureTerrain,
                    (priorTerrain * priorPressure), priorK);

            // use conditional probability formula on assignment sheet to get what
            // we want (but what is it that we want....)
            double[] kProbs = new double[ProblemSpec.CAR_MOVE_RANGE];
            double kProbsSum = 0;
            double kProb;
            for (int k = 0; k < ProblemSpec.CAR_MOVE_RANGE; k++) {
                kProb = magicFormula(pCarGivenK[k], pDriverGivenK[k],
                        pTireGivenK[k], pPressureTerrainGivenK[k], priorK);
                kProbsSum += kProb;
                kProbs[k] = kProb;
            }

            // Normalize
            for (int k = 0; k < ProblemSpec.CAR_MOVE_RANGE; k++) {
                kProbs[k] /= kProbsSum;
            }

            return kProbs;
        }

        // Copied from support code
        private double[] convertSlipProbs(double slipProb, TirePressure pressure) {

            // Adjust slip probability based on tire pressure
            if (pressure == TirePressure.SEVENTY_FIVE_PERCENT) {
                slipProb *= 2;
            } else if (pressure == TirePressure.ONE_HUNDRED_PERCENT) {
                slipProb *= 3;
            }
            // Make sure new probability is not above max
            if (slipProb > ProblemSpec.MAX_SLIP_PROBABILITY) {
                slipProb = ProblemSpec.MAX_SLIP_PROBABILITY;
            }

            // for each terrain, all other action probabilities are uniform over
            // remaining probability
            double[] kProbs = new double[ProblemSpec.CAR_MOVE_RANGE];
            double leftOver = 1 - slipProb;
            double otherProb = leftOver / (ProblemSpec.CAR_MOVE_RANGE - 1);
            for (int i = 0; i < ProblemSpec.CAR_MOVE_RANGE; i++) {
                if (i == ps.getIndexOfMove(ProblemSpec.SLIP)) {
                    kProbs[i] = slipProb;
                } else {
                    kProbs[i] = otherProb;
                }
            }

            return kProbs;
        }

        // Copied from support code
        private double[] bayesRule(double[] condProb, double priorA, double priorB) {

            double[] swappedProb = new double[condProb.length];

            for (int i = 0; i < condProb.length; i++) {
                swappedProb[i] = (condProb[i] * priorA) / priorB;
            }
            return swappedProb;
        }

        // Copied from support code
        private double magicFormula(double pA, double pB, double pC, double pD,
                                    double priorE) {
            return pA * pB * pC * pD * priorE;
        }

        // Given certain terrain, return the best specification in that terrain
        // return a [car, driver, tire, tire]
        public Object[] getBestSpecificationGivenTerrain(Terrain terrain) {
            List<TirePressure> validPressures = new ArrayList<>();
            validPressures.add(TirePressure.FIFTY_PERCENT);
            validPressures.add(TirePressure.SEVENTY_FIVE_PERCENT);
            validPressures.add(TirePressure.ONE_HUNDRED_PERCENT);

            double bestExpect = -1.0;
            Object[] rt = new Object[4];

            // If we want to reach the goal in MaxT timesteps, each step we should move N/MaxT cells.
            double unitTimeDistance = (double) ps.getN() / ps.getMaxT();

            // move Distance: -4..5
            double[] moveDistance = new double[12];
            for (int i = 0; i < 10; i++) {
                moveDistance[i] = ProblemSpec.CAR_MIN_MOVE + i;
            }

            // Slip and breakdown takes T timesteps to recover, in T timesteps we are expected move T*unitTimeDistance,
            //   but we didn't, so set the move distance to the negation of that distance
            moveDistance[10] = -ps.getSlipRecoveryTime() * unitTimeDistance;
            moveDistance[11] = -ps.getRepairTime() * unitTimeDistance;

            System.out.println(terrain);

            for (String car : ps.getCarOrder()) {
                for (String driver : ps.getDriverOrder()) {
                    for (Tire tire : ps.getTireOrder()) {
                        for (TirePressure pressure : validPressures) {
                            double[] moveProbs = getMoveProbs(terrain, car, driver, tire, pressure);
                            double expect = 0.0;
                            for (int i = 0; i < ProblemSpec.CAR_MOVE_RANGE; i++) {
                                expect += moveProbs[i] * moveDistance[i];
                            }
                            System.out.print(car + " " + driver + " " + tire + " " + pressure + " " + expect + "\n");
                            if (expect > bestExpect) {
                                bestExpect = expect;
                                rt = new Object[]{car, driver, tire, pressure, expect};
                            }
                        }
                    }
                }
            }

            System.out.println("======================================================");

            return rt;
        }

        public Map<Terrain, Object[]> getBestSpecification() {
            Map<Terrain, Object[]> map = new HashMap<>();
            for (Terrain terrain : ps.terrainOrder) {
                Object[] spec = getBestSpecificationGivenTerrain(terrain);
                map.put(terrain, spec);
            }
            return map;
        }

    }

    // In a state node with all the possible action performed (# of possible actions == # of children).
    // Perform UCT to select the best node to visit.
    public StateNode uct(StateNode current) throws Exception {
        Action bestAction = null;
        double bestScore = -1.0;
        int N = current.getVisit();

        // Iterate over all possible actions
        for (Action action : current.getMap().keySet()) {
            ActionNode actionNode = current.getMap().get(action);
            double reward = actionNode.getReward();
            int n_i = actionNode.getVisit();

            // w_i/n_i + c * \sqrt{ln N_i / n_i}
            // c is set to \sqrt{2} as recommended by the Wikipedia
            double score = (reward / n_i) + Math.sqrt(2 * Math.log(N) / n_i);
            if (score > bestScore) {
                bestAction = action;
                bestScore = score;
            }
        }

        // Initialize mySimulator, set timestep and current state
        MySimulator simulator = new MySimulator(ps);
        simulator.reset();
        simulator.setSteps(current.getTimestep());
        simulator.setCurrentState(current.getStateTuple().state);

        // mySimulator will return next stateTuple given current state and action
        StateTuple nextStateTuple = simulator.myStep(bestAction);
        StateNode nextNode = current.getMap().get(bestAction).getMap().get(nextStateTuple);
        if (nextNode == null) {
            throw new Exception("Null Pointer");
        }
        return nextNode;
    }

    // Selection phase of MCTS
    // Travelling down the Monte Carlo Search Tree to select a node for the Exapnsion phase,
    //   or if a terminal node is returned, directly perform backpropagation.

    // 1. current node is a terminal node (WIN or LOSE), return current node
    // 2. current node has at least one UNVISITED action, return current node
    // 3. current node has visited all the possible actions, using UCT to select the
    //      best node to select, move to the best node, keep running Selection until
    //      current node satisfies 1. or 2. Return that node.

    public StateNode Selection(StateNode stateNode) throws Exception {
        // actually we just need to stop when we reach a node whose number of children
        //   is less than number of possible action. (Terminal node has 0 children and
        //   it also satisfies the upper condition)
        int numAllActions = helper.getNumPossibleActions(stateNode.stateTuple);
        int numVisitedActions = stateNode.getMap().size();
        boolean terminate = numVisitedActions < numAllActions;

        StateNode node = stateNode;
        while (!terminate) {
            node = uct(node);
            numAllActions = helper.getNumPossibleActions(node.stateTuple);
            numVisitedActions = node.getMap().size();
            terminate = numVisitedActions < numAllActions;
        }

        return node;
    }

    // Expansion phase of MCTS
    // Given the stateNode with at least one UNVISITED action,
    //   select an UNVISITED action, add a child ActionNode,
    //   then add all the possible stateNode(s) as a child(ren) of stateNode.
    // Return a list of all next stateNodes
    public List<StateNode> Expansion(StateNode stateNode) {
        // Store all the information of current state
        int timesteps = stateNode.timestep;
        State state = stateNode.stateTuple.state;
        String carType = state.getCarType();
        int consumption = helper.getFuelConsumption(state);
        int fuel = state.getFuel();
        TirePressure tirePressure = state.getTirePressure();
        String driver = state.getDriver();
        Tire tire = state.getTireModel();

        State nextState;
        StateTuple nextTuple;
        StateNode nextNode;

        Set<Action> availableActions = helper.getAllPossibleActions(ps, stateNode.stateTuple);
        Set<Action> visitedActions = stateNode.getMap().keySet();
        Action chosenAction = null;
        for (Action action : availableActions) {
            if (!visitedActions.contains(action)) {
                chosenAction = action;
                break;
            }
        }
        assert chosenAction != null;

        List<StateNode> nodes = new ArrayList<>();

        // Action Node
        ActionNode actionNode = new ActionNode(chosenAction, stateNode);
        stateNode.getMap().put(chosenAction, actionNode);

        if (chosenAction.getActionType() == ActionType.MOVE) {
            // Consume fuel if level > 1
            if (ps.getLevel().getLevelNumber() > 1) {
                fuel -= consumption;
            }
            // MOVE -4 .. 5
            for (int i = ProblemSpec.CAR_MIN_MOVE; i <= ProblemSpec.CAR_MAX_MOVE; i++) {

                int pos = state.getPos() + i;
                if (1 <= pos && pos <= ps.getN()) {
                    nextState = new State(pos, false, false, carType,
                            fuel, tirePressure, driver, tire);

                    nextTuple = new StateTuple(nextState, false, false);
                    nextNode = new StateNode(nextTuple, actionNode, timesteps + 1);
                    nodes.add(nextNode);

                    // Update Action Node's Map
                    actionNode.getMap().put(nextTuple, nextNode);
                }
            }

            nextState = new State(state.getPos(), false, false, carType, fuel, tirePressure, driver, tire);
            // MOVE Slip
            nextTuple = new StateTuple(nextState, true, false);
            nextNode = new StateNode(nextTuple, actionNode, timesteps + ps.getSlipRecoveryTime());
            nodes.add(nextNode);
            actionNode.getMap().put(nextTuple, nextNode);
            // Move Breakdown
            nextTuple = new StateTuple(nextState, false, true);
            nextNode = new StateNode(nextTuple, actionNode, timesteps + ps.getRepairTime());
            nodes.add(nextNode);
            actionNode.getMap().put(nextTuple, nextNode);
        } else {
            MySimulator simulator = new MySimulator(ps);
            simulator.reset();
            simulator.setCurrentState(stateNode.stateTuple.state);
            simulator.setSteps(stateNode.timestep);

            nextTuple = simulator.myStep(chosenAction);
            nextNode = new StateNode(nextTuple, actionNode, simulator.getSteps());
            nodes.add(nextNode);
            actionNode.getMap().put(nextTuple, nextNode);
        }

        return nodes;
    }

    // Simulation(roll-out) phase of MCTS
    // initialize a simulator, and set simulator's timesteps and state
    // use random policy to select action when roll-out
    // return the reward
    public double Simulation(StateNode stateNode) {
        int start = stateNode.stateTuple.state.getPos();
        MySimulator simulator = new MySimulator(ps);

        simulator.reset();
        simulator.setSteps(stateNode.getTimestep());
        simulator.setCurrentState(stateNode.stateTuple.state);

        StateTuple tuple = stateNode.stateTuple;

        outer:
        while (simulator.getSteps() < ps.getMaxT()) {
            State state = tuple.state;

            Terrain terrain = ps.getEnvironmentMap()[state.getPos() - 1];
            // bestSpec = [car, driver, tire, tirePressure, expectedMoveDistance]
            Object[] bestSpec = terrain2spec.get(terrain);
            String bestCar = (String) bestSpec[0];
            String bestDriver = (String) bestSpec[1];
            Tire bestTire = (Tire) bestSpec[2];
            TirePressure bestPressure = (TirePressure) bestSpec[3];

            if (!state.getCarType().equals(bestCar)) {
                Action action = new Action(ActionType.CHANGE_CAR, bestCar);
                tuple = simulator.myStep(action);
                if (simulator.getSteps() >= ps.getMaxT()) {
                    break;
                }
            }

            if (!state.getDriver().equals(bestDriver)) {
                Action action = new Action(ActionType.CHANGE_DRIVER, bestDriver);
                tuple = simulator.myStep(action);
                if (simulator.getSteps() >= ps.getMaxT()) {
                    break;
                }
            }

            if (!state.getTireModel().equals(bestTire)) {
                Action action = new Action(ActionType.CHANGE_TIRES, (Tire) bestSpec[2]);
                tuple = simulator.myStep(action);
                if (simulator.getSteps() >= ps.getMaxT()) {
                    break;
                }
            }

            int consumption = helper.getFuelConsumption(tuple.state);
            // Add fuel if not enough fuel
            while (tuple.state.getFuel() < consumption) {
                Action action = new Action(ActionType.ADD_FUEL, 10);
                tuple = simulator.myStep(action);
                if (simulator.getSteps() >= ps.getMaxT()) {
                    break outer;
                }
            }

            Action action = new Action(ActionType.MOVE);
            tuple = simulator.myStep(action);
            if (tuple.state.getPos() == ps.getN()) {
                break outer;
            }

        }
        int end = tuple.state.getPos();
        int timesteps = simulator.getSteps();
        return helper.calculateReward(start, end, timesteps);
    }

    public double Simulation_level4(StateNode stateNode) {
        int start = stateNode.stateTuple.state.getPos();
        MySimulator simulator = new MySimulator(ps);

        simulator.reset();
        simulator.setSteps(stateNode.getTimestep());
        simulator.setCurrentState(stateNode.stateTuple.state);

        StateTuple tuple = stateNode.stateTuple;

        outer:
        while (simulator.getSteps() < ps.getMaxT()) {
            State state = tuple.state;

            Terrain terrain = ps.getEnvironmentMap()[state.getPos() - 1];
            // bestSpec = [car, driver, tire, tirePressure, expectedMoveDistance]
            Object[] bestSpec = terrain2spec.get(terrain);
            String bestCar = (String) bestSpec[0];
            String bestDriver = (String) bestSpec[1];
            Tire bestTire = (Tire) bestSpec[2];
            TirePressure bestPressure = (TirePressure) bestSpec[3];

            if (!state.getCarType().equals(bestCar) || !state.getDriver().equals(bestDriver)) {
                Action action = new Action(ActionType.CHANGE_CAR_AND_DRIVER, bestCar, bestDriver);
                tuple = simulator.myStep(action);
                if (simulator.getSteps() >= ps.getMaxT()) {
                    break;
                }
            }

            if (!state.getTireModel().equals(bestTire)) {
                Action action = new Action(ActionType.CHANGE_TIRES, (Tire) bestSpec[2]);
                tuple = simulator.myStep(action);
                if (simulator.getSteps() >= ps.getMaxT()) {
                    break;
                }
            }

            int consumption = helper.getFuelConsumption(tuple.state);
            // Add fuel if not enough fuel
            while (tuple.state.getFuel() < consumption) {
                Action action = new Action(ActionType.ADD_FUEL, 10);
                tuple = simulator.myStep(action);
                if (simulator.getSteps() >= ps.getMaxT()) {
                    break outer;
                }
            }

            Action action = new Action(ActionType.MOVE);
            tuple = simulator.myStep(action);
            if (tuple.state.getPos() == ps.getN()) {
                break outer;
            }

        }
        int end = tuple.state.getPos();
        int timesteps = simulator.getSteps();
        return helper.calculateReward(start, end, timesteps);
    }

    public double Simulation_level5(StateNode stateNode) {
        int start = stateNode.stateTuple.state.getPos();
        MySimulator simulator = new MySimulator(ps);

        simulator.reset();
        simulator.setSteps(stateNode.getTimestep());
        simulator.setCurrentState(stateNode.stateTuple.state);

        StateTuple tuple = stateNode.stateTuple;

        outer:
        while (simulator.getSteps() < ps.getMaxT()) {
            State state = tuple.state;

            Terrain terrain = ps.getEnvironmentMap()[state.getPos() - 1];
            // bestSpec = [car, driver, tire, tirePressure, expectedMoveDistance]
            Object[] bestSpec = terrain2spec.get(terrain);
            String bestCar = (String) bestSpec[0];
            String bestDriver = (String) bestSpec[1];
            Tire bestTire = (Tire) bestSpec[2];
            TirePressure bestPressure = (TirePressure) bestSpec[3];

            if (!state.getCarType().equals(bestCar) || !state.getDriver().equals(bestDriver)) {
                Action action = new Action(ActionType.CHANGE_CAR_AND_DRIVER, bestCar, bestDriver);
                tuple = simulator.myStep(action);
                if (simulator.getSteps() >= ps.getMaxT()) {
                    break;
                }
            }

            if (!state.getTireModel().equals(bestTire)) {
                Action action = new Action(ActionType.CHANGE_TIRE_FUEL_PRESSURE, (Tire) bestSpec[2], 10, state.getTirePressure());
                tuple = simulator.myStep(action);
                if (simulator.getSteps() >= ps.getMaxT()) {
                    break;
                }
            }

            int consumption = helper.getFuelConsumption(tuple.state);
            // Add fuel if not enough fuel
            while (tuple.state.getFuel() < consumption) {
                Action action = new Action(ActionType.ADD_FUEL, 10);
                tuple = simulator.myStep(action);
                if (simulator.getSteps() >= ps.getMaxT()) {
                    break outer;
                }
            }

            Action action = new Action(ActionType.MOVE);
            tuple = simulator.myStep(action);
            if (tuple.state.getPos() == ps.getN()) {
                break outer;
            }

        }
        int end = tuple.state.getPos();
        int timesteps = simulator.getSteps();
        return helper.calculateReward(start, end, timesteps);
    }

    // Backpropagation phase of MCTS
    // Perform backpropagation from leaf to root
    public void backPropagation(StateNode node, double result) {
        StateNode cur = node;

        while (cur.parent != null) {
            cur.increaseVisit();
            cur.increaseReward(result);

            ActionNode parentAction = cur.getParent();
            parentAction.increaseVisit();
            parentAction.increaseReward(result);

            cur = parentAction.getParent();
        }

        cur.increaseVisit();
        cur.increaseReward(result);
    }

    // Perform a single simulation of MCTS
    // return array of double containing {sum of rewards, # of nodes created in Expansion phase}
    public double[] simulate(StateNode root) throws Exception {
        StateNode selectedNode = Selection(root);
        double rt = 0;
        // If selected node is a terminal node, directly perform back propagation
        if (selectedNode.getGameState() != GameState.RUNNING) {
            int start = root.stateTuple.state.getPos();
            int end = selectedNode.stateTuple.state.getPos();
            int timesteps = selectedNode.getTimestep();
            double result = helper.calculateReward(start, end, timesteps);
            backPropagation(selectedNode, result);
            rt += result;
            return new double[]{rt, 1.0};
        } else {
            List<StateNode> nodes = Expansion(selectedNode);
            for (StateNode node : nodes) {
                double result;
                if (ps.getLevel().getLevelNumber() == 4) {
                    result = Simulation_level4(node);
                } else if (ps.getLevel().getLevelNumber() == 5) {
                    result = Simulation_level5(node);
                } else {
                    result = Simulation(node);
                }
                backPropagation(node, result);
                rt += result;
            }
            return new double[]{rt, nodes.size()};
        }
    }

    public void solve(String outputPath) throws Exception {
        State initState = State.getStartState(ps.getFirstCarType(), ps.getFirstDriver(), ps.getFirstTireModel());
        StateNode root = new StateNode(new StateTuple(initState, false, false), null, 0); // initialization of timestep is 0
        Simulator writer = new Simulator(ps, outputPath, true);
        writer.reset();

        StateTuple nextStateTuple = null;
        int i = 0;

        while (i < ps.getMaxT()) {
            long start = System.currentTimeMillis();
            double elapsed = (System.currentTimeMillis() - start) / 1000.0;
            int count = 0;
            int win = 0;
            // Keep simulating for 15 sec
            while (elapsed < 1) {
                double[] vals = simulate(root);
                count += vals[1];
                win += vals[0] >= 1 ? 1 : 0;
                elapsed = (System.currentTimeMillis() - start) / 1000.0;
            }

            System.out.println("Simulated " + count + " times.. " + win + " win..");

            // Select the best Action
            double ratio = -1.0;
            Action bestAction = null;
            for (Action action : root.getMap().keySet()) {
                ActionNode actionNode = root.getMap().get(action);
                if (actionNode.getRatio() > ratio) {
                    bestAction = action;
                    ratio = actionNode.getRatio();
                }
            }
            System.out.println("The best action is : " + bestAction);
            if (bestAction == null) {
                bestAction = new Action(ActionType.MOVE);
            }

            nextStateTuple = writer.myStep(bestAction);
            if (nextStateTuple != null) {
                System.out.println(nextStateTuple);
                System.out.println("==================");
            }
            root = root.getMap().get(bestAction).getMap().get(nextStateTuple);
            root.parent = null;

            i = root.getTimestep();

            if (writer.isGoalState(nextStateTuple.state)) {
                System.out.println("GOOOOOOAAAAAL!!");
                break;
            }

        }

        if (!writer.isGoalState(nextStateTuple.state)) {
            System.out.println("Fail!!");

        }

    }

    public static void main(String[] args) {

        String inputPath = "examples/level_5/input_lvl5_2.txt";
        String outputPath = "examples/level_5/output_lvl5_2.txt";

        ProblemSpec ps;
        try {
            ps = new ProblemSpec(inputPath);
            System.out.println(ps.terrainOrder);
            MCTS treeSearch = new MCTS(ps);
            treeSearch.solve(outputPath);
        } catch (IOException e) {
            System.exit(1);
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

}
