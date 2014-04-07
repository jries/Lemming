package org.lemming.tests;


import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Frame;
import org.lemming.data.QueueStore;
import org.lemming.inputs.ImageJTIFFLoader;
import org.lemming.utils.LemMING;
import org.lemming.utils.MatlabOpener;

public class MatlabOpenerTest {
	MatlabOpener o;
	QueueStore<Frame> frames;
	ImageJTIFFLoader j;
	
	@Before
	public void setUp() throws Exception {
		o = new MatlabOpener("C:/Users/Ries/Desktop/joran/testSTORM/testSTORM_v1/","testSTORM"); 
		frames = new QueueStore<Frame>();
		j = new ImageJTIFFLoader("C:/Users/Ries/Desktop/joran/testSTORM/testSTORM_v1/created_img/tdffghdcst.tif");
	}

	@Test
	public void testSTORM() throws InterruptedException {
		new Thread(o).start();
	
		while (!o.isUserDone()) {
			LemMING.pause(100);
		}	

		new Thread(j).start();
		
		while (j.hasMoreOutputs()) {
			LemMING.pause(100);
		}	
		
		j.show();
		
		LemMING.pause(5000);
	}
	
}
