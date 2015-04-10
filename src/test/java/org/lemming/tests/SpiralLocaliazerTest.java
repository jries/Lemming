package org.lemming.tests;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.QueueStore;
import org.lemming.interfaces.Localization;
import org.lemming.interfaces.Store;
import org.lemming.outputs.GaussRenderOutput;
import org.lemming.utils.LemMING;

public class SpiralLocaliazerTest {

	@Before
	public void setUp() throws Exception {
	}

	@Test
	public void test() {
		
		SpiralLocalizer sl = new SpiralLocalizer();
		
		GaussRenderOutput gr = new GaussRenderOutput();
		
		Store<Localization> store = new QueueStore<Localization>();
		
		sl.setOutput(store);
		gr.setInput(store);
		
		new Thread(sl).start();
		new Thread(gr).start();
		
		LemMING.pause(10000);
		
	}

}
