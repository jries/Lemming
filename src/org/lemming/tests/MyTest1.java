package org.lemming.tests;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Localization;
import org.lemming.queue.QueueStore;
import org.lemming.queue.Store;
import org.lemming.data.XYLocalization;
import org.lemming.inputs.RandomLocalizer;
import org.lemming.outputs.PrintToScreen;
import org.lemming.utils.LemMING;

public class MyTest1 {
		
	@Before
	public void setUp() throws Exception {
		
	}

	@Test
	public void test() {
		PrintToScreen ps = new PrintToScreen();
		ps.process(new XYLocalization(1, 2));
		
                RandomLocalizer rl = new RandomLocalizer(100, 256, 256);
                while (rl.hasMoreOutputs()) {
                    ps.process(rl.newOutput());
                }
	}
	
	

}
