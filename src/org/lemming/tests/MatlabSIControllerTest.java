package org.lemming.tests;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Localization;
import org.lemming.data.QueueStore;
import org.lemming.inputs.RandomLocalizer;
import org.lemming.matlab.MatlabSIController;
import org.lemming.utils.LemMING;

public class MatlabSIControllerTest {

	QueueStore<Localization> localizations;
	RandomLocalizer rl;
	
	MatlabSIController<Localization> c;
	
	@Before
	public void setUp() throws Exception {
		
		localizations = new QueueStore<Localization>();

		// Initiate Random localizer	
		rl = new RandomLocalizer(500, 256, 256);
		rl.setOutput(localizations);
		
		// Initiate Matlab SI controller
		c = new MatlabSIController<Localization>();
		c.setInput(localizations);
	}

	@Test
	public void testSimpleImport() {
		
		new Thread(rl).start();
		new Thread(c).start();
		
		while (rl.hasMoreOutputs() || !localizations.isEmpty()) {
			LemMING.pause(100);
		}		
		
		LemMING.pause(2000);
	}
}


