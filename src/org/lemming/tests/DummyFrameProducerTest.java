package org.lemming.tests;

import static org.junit.Assert.*;
import net.imglib2.type.numeric.integer.UnsignedShortType;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Frame;
import org.lemming.data.ImgLib2Frame;
import org.lemming.data.QueueStore;
import org.lemming.dummy.DummyFrameProducer;
import org.lemming.utils.LemMING;

/**
 * Test class for inserting dummy frames into a Store 
 * 
 * @author Joe Borbely, Thomas Pengo
 */
public class DummyFrameProducerTest {

	DummyFrameProducer d;
	
	@Before
	public void setUp() throws Exception {
		d = new DummyFrameProducer();
	}

	@Test
	public void test() {		
                int frame_count = 0;
                while (d.hasMoreOutputs()) {
                    d.newOutput();
                    ++frame_count;
                }
		assertEquals(frame_count, 100);
	}

}
