package org.lemming.tests;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.QueueStore;
import org.lemming.data.XYLocalization;
import org.lemming.inputs.RandomLocalizer;
import org.lemming.interfaces.Localization;
import org.lemming.interfaces.Store;
import org.lemming.outputs.PrintToScreen;
import org.lemming.utils.LemMING;

public class MyTest1 {
		
	@Before
	public void setUp() throws Exception {
		
	}

	@Test
	public void test() {
		
		Store<Localization> s = new QueueStore<Localization>();
		
		s.put(new XYLocalization(1, 2));
		
		RandomLocalizer rl = new RandomLocalizer(100, 256, 256);
		rl.setOutput(s);
		
		PrintToScreen ps = new PrintToScreen();
		ps.setInput(s);
		
		new Thread(rl).start();
		new Thread(ps).start();
		
		LemMING.pause(1000);
	}
	
	

}
