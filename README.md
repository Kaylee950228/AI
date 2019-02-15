
**Problem Package**

The main problem class and parser.

To use it simply create a new ProblemSpec object and pass in the path to the input file of your choosing as an argument.

```$xslt
ProblemSpec ps = new ProblemSpec("path/to/inputFile.txt");
```

Enjoy the problem instance object

**Simulator package**

The simulator for you to test your policies and for use when we assess your implementation.
The simulator class will **automatically output the steps taken to the output file** provided when a terminal state is reached (i.e. the goal is reached or max timesteps is reached).

Initialize the simulator by either:
 1. passing your problem spec along with output file path to its constructor
 2. passing the input file path along with output file path to its constructor 

```$xslt
ProblemSpec ps = new ProblemSpec("path/to/inputFile.txt");
Simulator sim = new Simulator(ps, "path/to/outputFile.txt");
// or 
Simulator sim = new Simulator("path/to/inputFile.txt", "path/to/outputFile.txt");
```

To utilize the simulator, simply call the step function which accepts an
Action class instance and returns the next state.
You can also call the getSteps method to get the current number of steps for the simulation.

```$xslt
Action a = new Action(ActionType.MOVE)    // or some other action
State nextState = sim.step(a);
int stepsTakenSoFar = sim.getSteps();
```


