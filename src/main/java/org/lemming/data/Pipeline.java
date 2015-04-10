package org.lemming.data;

import java.util.LinkedList;

import org.lemming.interfaces.Processor;
import org.lemming.interfaces.Source;
import org.lemming.utils.LemMING;

/**
 * @author Ronny Sczech
 *
 */
public class Pipeline implements Runnable {
	
	private ThreadGroup group;
	private LinkedList<Thread> pipe;

	/**
	 * 
	 */
	public Pipeline(){
		this.group = new ThreadGroup("Pipe");
		this.pipe = new LinkedList<Thread>();
	}
	
	/**
	 * @param module - Runnable to add
	 */
	public void add(Runnable module){
		pipe.addLast(new Thread(group,module,module.getClass().getName()));
	}
	
	/**
	 * @param module - Runnable to run as sequential
	 */
	@SuppressWarnings("rawtypes")
	public void addSequential(Runnable module){
		if (!pipe.isEmpty()) run();
		module.run();
		if (module instanceof Source){
			Source lm = (Source) module;
			while(lm.hasMoreOutputs()){ //wait until store is filled up completely
				LemMING.pause(5);
			}
			System.out.println("module " + lm.getClass().getSimpleName() + " completed!");
		} else if (module instanceof Processor){
			Processor lm = (Processor) module;
			while(lm.hasMoreOutputs()){ //wait until store is filled up completely
				LemMING.pause(5);
			}
			System.out.println("module " + lm.getClass().getSimpleName() + " completed!");
		}
		pipe.clear();
	}

	@Override
	public void run() {
		group.setDaemon(true);
		for(Thread starter:pipe){
			starter.start();
		}
		for(Thread joiner:pipe){
			try {
				joiner.join();
			} catch (InterruptedException e) {
				System.err.println(e.getMessage());
			}
		}
		group.interrupt();
	}

}
