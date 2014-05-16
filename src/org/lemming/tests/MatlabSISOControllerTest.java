package org.lemming.tests;

import matlabcontrol.MatlabConnectionException;
import matlabcontrol.MatlabInvocationException;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Localization;
import org.lemming.data.QueueStore;
import org.lemming.inputs.RandomLocalizer;
import org.lemming.matlab.MatlabSISOController;
import org.lemming.outputs.GaussRenderOutput;
import org.lemming.utils.LemMING;

public class MatlabSISOControllerTest {

	QueueStore<Localization> localizations;
	QueueStore<Localization> localizations2;

	RandomLocalizer rl;
	MatlabSISOController<Localization,Localization> c;
	GaussRenderOutput gro;


	@Before
	public void setUp() throws Exception {
		
		localizations = new QueueStore<Localization>();
		localizations2 = new QueueStore<Localization>();

		// Initiate Random localizer	
		rl = new RandomLocalizer(500, 256, 256);
		rl.setOutput(localizations);
	}

	@Test
	public void testPALMsiever() {

		// Initiate MATLABCommunicator
		String dir = "C:/Users/Ries/Desktop/joran/palmsiever/PALM_DIR";
		try {
			c = new MatlabSISOController<Localization,Localization>("Localization",dir,"PALMsiever('X','Y')");
		} catch (MatlabConnectionException e) {
			e.printStackTrace();
		}

		// Initiate HistogramRender
		gro = new GaussRenderOutput();
		
		// Set pipeline
		c.setInput(localizations);  
		c.setOutput(localizations2);
		gro.setInput(localizations2);
		
		new Thread(rl).start();
		new Thread(c).start();
		new Thread(gro).start();

		while (rl.hasMoreOutputs() || !localizations.isEmpty()) {
			LemMING.pause(100);
		}		
		while (!c.isUserDone() || !localizations2.isEmpty()) {
			LemMING.pause(100);
		}		

		LemMING.pause(2000);
	}
	
}
