package org.lemming.processors;

import net.imglib2.algorithm.MultiThreaded;
import net.imglib2.multithreading.SimpleMultiThreading;

import org.lemming.interfaces.Processor;
import org.lemming.interfaces.Store;
import org.lemming.outputs.NullStoreWarning;

/**
 * @author Ronny Sczech
 *
 * @param <T1> - data type
 * @param <T2> - data type
 */
@SuppressWarnings("deprecation")
public abstract class SingleInputSingleOutput<T1,T2> implements Runnable, Processor<T1, T2>, MultiThreaded {

	protected Store<T1> input;
	protected Store<T2> output;
	private volatile boolean running;
	private int numThreads;
	private Thread[] threads;
	private ThreadGroup group;

	
	/**
	 * 
	 */
	public SingleInputSingleOutput(){
		this.running = true;
		this.numThreads = 1;
		this.group = new ThreadGroup("Group");
	}

	@Override
	public void run() {
		
		if (input==null || output==null)
			throw new NullStoreWarning(this.getClass().getName()); 
		
		threads = new Thread[numThreads];
		for (int ithread = 0; ithread < threads.length; ithread++) {
			threads[ithread] = new Thread(group,(1+ithread)+"/"+threads.length) {
				
				@Override
				public void run() {
					while (running) {
						process(nextInput());
					}
				}
				
			};
		}
		
		SimpleMultiThreading.startAndJoin(threads);
	}
	
	/**
	 * Method to be overwritten by childs of this class.
	 * @param element - element
	 */
	public abstract void process(T1 element);
	
	T1 nextInput() {
		return input.get();
	}
	
	/**
	 * 
	 */
	public void stop(){
		running = false;
		group.interrupt();
		/*for (int ithread = 0; ithread < threads.length; ithread++) {
			threads[ithread].interrupt();
		}/*/
	}

	@Override
	public void setInput(Store<T1> s) {
		input = s;
	}

	@Override
	public void setOutput(Store<T2> s) {
		output = s;
	}
	
	@Override
	public int getNumThreads() {
		return numThreads;
	}

	@Override
	public void setNumThreads() {
		this.numThreads=Runtime.getRuntime().availableProcessors()-1;
	}

	@Override
	public void setNumThreads(int numThreads) {
		this.numThreads=numThreads;		
	}

}
