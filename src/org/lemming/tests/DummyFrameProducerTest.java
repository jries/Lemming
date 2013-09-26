package org.lemming.tests;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Frame;
import org.lemming.data.QueueStore;
import org.lemming.dummy.DummyFrameProducer;

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
		try {
			new Thread(d).start();
			
			Thread.sleep(1000);
			
			assertEquals(q.getLength(), 100);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}

}
