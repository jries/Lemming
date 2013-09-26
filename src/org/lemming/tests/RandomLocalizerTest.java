package org.lemming.tests;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Localization;
import org.lemming.data.QueueStore;
import org.lemming.input.RandomLocalizer;
import org.lemming.outputs.GaussRenderOutput;

public class RandomLocalizerTest {

	RandomLocalizer rl;
	GaussRenderOutput gro;
	QueueStore<Localization> localizations;
	
	@Before
	public void setUp() throws Exception {
		
		rl = new RandomLocalizer(5000, 256, 256);
		localizations = new QueueStore<Localization>();
		gro = new GaussRenderOutput(256, 256);
		
		rl.setOutput(localizations);
		gro.setInput(localizations);
	}

	@Test
	public void test() {
		new Thread(rl).start();
		new Thread(gro).start();
		while (true) {}		
	}
}
