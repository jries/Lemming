package org.lemming.tests;

import matlabcontrol.MatlabConnectionException;
import matlabcontrol.MatlabInvocationException;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Localization;
import org.lemming.data.QueueStore;
import org.lemming.inputs.MatlabSOController;
import org.lemming.outputs.GaussRenderOutput;
import org.lemming.utils.LemMING;

public class MatlabSOControllerTest {
	
	QueueStore<Localization> localizations;

	MatlabSOController<Localization> c;
	GaussRenderOutput gro;

	@Before
	public void setUp() throws Exception {		
		
		localizations = new QueueStore<Localization>();
		
	}
	
	@Test
	public void testSimpleLocGrabber() {				// Require x and y vectors to be already in matlab workspace 
		try {
			c = new MatlabSOController<Localization>("Localization");
		} catch (MatlabConnectionException | MatlabInvocationException e) {
			e.printStackTrace();
		}
		
		// Initiate HistogramRender
		gro = new GaussRenderOutput();
		
		// Set pipeline
		c.setOutput(localizations);
		gro.setInput(localizations);
		
		new Thread(c).start();
		new Thread(gro).start();

		while (!c.isUserDone() || !localizations.isEmpty()) {
			LemMING.pause(100);
		}		

		LemMING.pause(2000);
	}
	
}
