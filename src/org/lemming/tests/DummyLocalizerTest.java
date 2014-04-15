package org.lemming.tests;

import static org.junit.Assert.*;
import net.imglib2.type.numeric.integer.UnsignedShortType;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Frame;
import org.lemming.data.ImgLib2Frame;
import org.lemming.data.Localization;
import org.lemming.queue.QueueStore;
import org.lemming.dummy.DummyFrameProducer;
import org.lemming.dummy.DummyLocalizer;
import org.lemming.utils.LemMING;

/**
 * Test class for inserting dummy frames into a Store and dummy 
 * localizations into a Store.
 * 
 * @author Joe Borbely, Thomas Pengo
 */
public class DummyLocalizerTest {

	DummyLocalizer<UnsignedShortType, ImgLib2Frame<UnsignedShortType>> d;
	
	@Before
	public void setUp() throws Exception {
		d = new DummyLocalizer<>();
	}

	@Test
	public void test() {
		DummyFrameProducer i = new DummyFrameProducer();
		
                int localization_count = 0;
                while (i.hasMoreOutputs()) {
                    localization_count += d.process(i.newOutput()).size();
                }
		assertEquals(localization_count, 200);
	}

}
