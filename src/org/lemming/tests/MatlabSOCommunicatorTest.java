package org.lemming.tests;

import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Frame;
import org.lemming.data.QueueStore;
import org.lemming.inputs.MatlabSOCommunicator;
import org.lemming.utils.LemMING;


/**
 * Test class for opening a matlab program and import a tiff stack (example of testSTORM).
 * 
 * @author Joe Borbely, Thomas Pengo, Joran Deschamps
 */

public class MatlabSOCommunicatorTest {

	MatlabSOCommunicator com;
	QueueStore<Frame> frames;
	
	@Before
	public void setUp() throws Exception {
		com = new MatlabSOCommunicator("C:/Users/Ries/Desktop/joran/testSTORM/testSTORM_v1/","testSTORM","C:/Users/Ries/Desktop/joran/testSTORM/testSTORM_v1/created_img/tdfsfst.tif"); 
		frames = new QueueStore<Frame>();
		
		com.setOutput(frames);
	}

	@Test
	public void test() {
		com.run();
		assertEquals(com.getNumFrames(), frames.getLength());				
		com.show();
		LemMING.pause(5000);
	}
	
}
