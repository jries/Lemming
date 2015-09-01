package org.lemming.pipeline;

import java.util.LinkedList;
import org.lemming.interfaces.Store;

public class Pipeline implements Runnable {

	private ThreadGroup group;
	private LinkedList<Thread> pipe;
	public Store zipStore = new FastStore();
	/**
	 * 
	 */
	public Pipeline(String projectName){
		this.group = new ThreadGroup(projectName);
		this.pipe = new LinkedList<>();
	}
	
	/**
	 * @param module - Runnable to add
	 */
	public void add(AbstractModule module){
		pipe.addLast(new Thread(group, module, module.getClass().getSimpleName()));
	}		
	
	/**
	 * @param module - Module to run as sequential
	 */
	public void addSequential(AbstractModule module){
		if (!pipe.isEmpty()) run();
		module.run();
	
		while(module.isRunning()){ //wait until store is filled up completely
			pause(10);
		}
		pipe.clear();
	}

	@Override
	public void run() {
		
		for(Thread starter:pipe)
			starter.start();
		
		for(Thread joiner:pipe){
			try {
				joiner.join();
			} catch (InterruptedException e) {
				System.err.println(e.getMessage());
			}
		}
		group.interrupt();
	}
	
	private static void pause(long ms){
		try {
			Thread.sleep(ms);
		} catch (InterruptedException e) {
			System.err.println(e.getMessage());
		}
	}
	
	
}
