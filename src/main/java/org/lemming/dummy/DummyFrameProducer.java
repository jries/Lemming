package org.lemming.dummy;

//import java.util.concurrent.locks.ReentrantReadWriteLock;

//import java.util.concurrent.atomic.AtomicInteger;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
//import net.imglib2.multithreading.SimpleMultiThreading;
import net.imglib2.type.numeric.integer.UnsignedShortType;

import org.lemming.data.ImgLib2Frame;
import org.lemming.interfaces.Source;
import org.lemming.interfaces.Store;

//@SuppressWarnings("deprecation")
/**
 * @author Ronny Sczech
 *
 */
public class DummyFrameProducer implements Source<ImgLib2Frame<UnsignedShortType>> {

	/**
	 * output
	 */
	public Store<ImgLib2Frame<UnsignedShortType>> output;
	private int curSlice = 0;
	//private int numThreads = Runtime.getRuntime().availableProcessors();

	final private int maxSlices = 99;
	
	/**
	 * 
	 */
	public DummyFrameProducer() {
		super();
	}

	@Override
	public void setOutput(Store<ImgLib2Frame<UnsignedShortType>> store) {
		output = store;
	}
	
	class DummyFrame extends ImgLib2Frame<UnsignedShortType> {
		long ID;
		
		DummyFrame(long ID) {
			super(ID, 256, 256, ArrayImgs.unsignedShorts(new long[]{256,256}));
			
			this.ID = ID;
		}
		
		public long getFrameNumber() {
			return ID;
		}

		@Override
		public RandomAccessibleInterval<UnsignedShortType> getPixels() {
			return null;
		}

		@Override
		public int getWidth() {
			return 0;
		}

		@Override
		public int getHeight() {
			return 0;
		}
	}
	
		
	@Override
	public void run() {
		//final Thread[] threads = new Thread[numThreads];
		//final AtomicInteger ai = new AtomicInteger(0);
		//for (int ithread = 0; ithread < threads.length; ithread++) {
		//	threads[ithread] = new Thread((1+ithread)+"/"+threads.length) {
		//	
		//		@Override
		//		public void run() {
		//
		//			for (int i = ai.getAndIncrement(); i < 99; i = ai.getAndIncrement()) {
					while (hasMoreOutputs()){
						output.put(new DummyFrame(curSlice++));
					}
						
		//			}
		//			
		//		}
		//			
		//	};
		//}
		//SimpleMultiThreading.startAndJoin(threads);
		// set finisher
		DummyFrame lastFrame =	new DummyFrame(curSlice);
		lastFrame.setLast(true);
		output.put(lastFrame);
		System.out.println("*** DummyFrameProducer ****");
	}	
			
	@Override
	public boolean hasMoreOutputs() {
		return curSlice < maxSlices;
	}

}
