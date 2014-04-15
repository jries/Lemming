package org.lemming.tests;

import static org.junit.Assert.*;
import net.imglib2.type.numeric.integer.UnsignedShortType;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.ImgLib2Frame;
import org.lemming.data.Localization;
import org.lemming.queue.QueueStore;
import org.lemming.dummy.DummyFrameProducer;
import org.lemming.dummy.DummyLocalizer;
import org.lemming.outputs.PrintToScreen;
import org.lemming.utils.LemMING;

/**
 * Test class for reading a Store of localizations and then writing the 
 * localizations to the screen.
 * 
 * @author Joe Borbely, Thomas Pengo
 */
public class PrintToScreenTest {

	PrintToScreen p;
	
	@Before
	public void setUp() throws Exception {
		p = new PrintToScreen();
	}

	@Test
	public void test() {
		DummyFrameProducer i = new DummyFrameProducer();
		DummyLocalizer<UnsignedShortType, ImgLib2Frame<UnsignedShortType>> d = new DummyLocalizer<>();
		
                while (i.hasMoreOutputs()) {
                    for (Localization l : d.process(i.newOutput())) {
                        p.process(l);
                    }
                }
	}

}
