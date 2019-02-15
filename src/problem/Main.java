package problem;

import simulator.Simulator;
import simulator.State;

import java.io.IOException;

public class Main {

    public static void main(String[] args) throws InterruptedException {
        long start = System.currentTimeMillis();
        Thread.sleep(15000);
        long end = System.currentTimeMillis();
        double elapsed = (end - start) / 1000.0;
        System.out.println(elapsed);

    }
}
