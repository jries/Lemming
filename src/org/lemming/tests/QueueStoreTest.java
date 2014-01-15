package org.lemming.tests;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.QueueStore;

/**
 * Test class for a Store.
 * 
 * @author Joe Borbely, Thomas Pengo
 *
 */
public class QueueStoreTest {

	QueueStore<Integer> q;
	
	@Before
	public void setUp() throws Exception {
		q = new QueueStore<>();
	}

	@Test
	public void testPut() {
		q.put(1);
		
		assertEquals(q.getLength(), 1);
	}

	@Test
	public void testGet() {
		q.put(1);
		
		assertEquals(q.get(), (Integer) 1);
	}

	@Test
	public void testFIFO() {
		q.put(1);
		q.put(2);
		
		assertEquals(q.get(), (Integer) 1);
	}

	@Test
	public void testBlockingBehaviour() {
		try{
			new Thread() {
				@Override
				public void run() {
					getQueue().get();
					
					getQueue().put(2);
				}
				
			}.start();
			
			Thread.sleep(1000);
			
			q.put(1);
			
			Thread.sleep(1000);
			
			assertEquals(q.get(), (Integer) 2); 
			
		} catch(Exception e) {
			fail("Should've not given me an exception!!");
		}
	}
	
	private QueueStore<Integer> getQueue() {
		return q;
	}
}
