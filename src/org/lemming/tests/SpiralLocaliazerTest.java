package org.lemming.tests;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Localization;
import org.lemming.queue.QueueStore;
import org.lemming.queue.Store;
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
		
                while (sl.hasMoreOutputs()) {
                        gr.process(sl.newOutput());
                }
	}

}
