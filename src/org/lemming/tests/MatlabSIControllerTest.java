package org.lemming.tests;

import matlabcontrol.MatlabConnectionException;
import matlabcontrol.MatlabInvocationException;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Localization;
import org.lemming.data.QueueStore;
import org.lemming.data.XYFLocalization;
import org.lemming.inputs.DriftXYFLocalizer;
import org.lemming.inputs.RandomLocalizer;
import org.lemming.matlab.MatlabSIController;
import org.lemming.utils.LemMING;

public class MatlabSIControllerTest {

	QueueStore<Localization> localizations;
	QueueStore<XYFLocalization> xyflocalizations;
	RandomLocalizer rl;
	DriftXYFLocalizer r2;

	MatlabSIController<Localization> c1;
	MatlabSIController<XYFLocalization> c2;
	
	@Before
	public void setUp() throws Exception {

		localizations = new QueueStore<Localization>();
		xyflocalizations = new QueueStore<XYFLocalization>();

		// Initiate Random localizer	
		rl = new RandomLocalizer(500, 256, 256);
		rl.setOutput(localizations);
		
		// Initiate RandomXYF localizer	
		r2 = new DriftXYFLocalizer(60000);
		r2.setOutput(xyflocalizations);

	}

	@Test
	public void testSimpleImport() throws MatlabInvocationException, MatlabConnectionException {
		c1 = new MatlabSIController<Localization>();
		c1.setInput(localizations);
		
		new Thread(rl).start();
		new Thread(c1).start();
		
		while (rl.hasMoreOutputs() || !localizations.isEmpty()) {
			LemMING.pause(100);
		}		
		
		LemMING.pause(2000);
		
		//Can put lines to assert if the data received in matlab is equal to what was sent (now only by eye)
	}

	@Test
	public void testSimpleXYFImport() throws MatlabConnectionException, MatlabInvocationException {

		c2 = new MatlabSIController<XYFLocalization>();
		c2.setInput(xyflocalizations);

		new Thread(r2).start();
		new Thread(c2).start();
		
		while (r2.hasMoreOutputs() || !xyflocalizations.isEmpty()) {
			LemMING.pause(100);
		}		
		
		LemMING.pause(2000);
	}
}


