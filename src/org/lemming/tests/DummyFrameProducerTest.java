package org.lemming.tests;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Frame;
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
	QueueStore<Frame> q;
	
	@Before
	public void setUp() throws Exception {
		d = new DummyFrameProducer();
		q = new QueueStore<Frame>();
		
		d.setOutput(q);
	}

	@Test
	public void test() {		
		new Thread(d).start();
		
		LemMING.pause(1000);
		
		assertEquals(q.getLength(), 100);
	}

}
