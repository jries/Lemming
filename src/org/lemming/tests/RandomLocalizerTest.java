package org.lemming.tests;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Localization;
import org.lemming.inputs.RandomLocalizer;
import org.lemming.interfaces.Rendering;
import org.lemming.outputs.GaussRenderOutput;
import org.lemming.outputs.HistogramRender;
//import org.lemming.outputs.Jzy3dScatterplot;
import org.lemming.utils.LemMING;

/**
 * Test class for creating randomly-positioned localizations and 
 * adding the localizations into a Store.
 * 
 * @author Joe Borbely, Thomas Pengo
 */
public class RandomLocalizerTest {

	RandomLocalizer rl;
	Rendering ren;
	
	@Before
	public void setUp() throws Exception {
		rl = new RandomLocalizer(50000, 256, 256);
	}

	@Test
	public void testGaussRender() {
		ren = new GaussRenderOutput(256, 256);
                rl.beforeRun();
                ren.beforeRun();
                while (rl.hasMoreOutputs()) {
                        ren.process(rl.newOutput());
                }
                ren.afterRun();
                rl.afterRun();
	}

	@Test
	public void testHistoRender() {
		ren = new HistogramRender();
                rl.beforeRun();
                ren.beforeRun();
                while (rl.hasMoreOutputs()) {
                        ren.process(rl.newOutput());
                }
                ren.afterRun();
                rl.afterRun();
	}

	@Test
	public void testMultiplHistoRender() {
		ren = new HistogramRender();
		Rendering histo2 = new HistogramRender();
                rl.beforeRun();
                ren.beforeRun();
                while (rl.hasMoreOutputs()) {
                        Localization localization = rl.newOutput();
                        ren.process(localization);
                        histo2.process(localization);
                }
                ren.afterRun();
                rl.afterRun();
	}
	
	/*
	@Test
	public void testScatterPlot() {
		ren = new Jzy3dScatterplot();
		ren.setInput(localizations);
		
		new Thread(rl).start();
		
		ren.run();
		
		LemMING.pause(10000);
	}
	*/

}
