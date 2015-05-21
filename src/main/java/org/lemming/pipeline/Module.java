package org.lemming.pipeline;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

public abstract class Module extends AbstractModule{
	
	public Module(){
	}
	
	@Override
	public void run() {

		if (!inputs.isEmpty()){ // first check for existing inputs
			beforeRun();
			
			final ArrayList< Future< Void > > futures = new ArrayList< >();
			
			for ( int taskNum = 0; taskNum < getNumThreads(); ++taskNum ){
	
				final Callable< Void > r = new Callable< Void >(){
					
					@Override
					public Void call() {
						while (running) {
								if (Thread.currentThread().isInterrupted()) break;
								Map<String, Element> data = nextInput();
								process(data);
						}
						return null;
					}
					
				};
				futures.add( service.submit( r ) );
			}
			
			for ( final Future< Void > f : futures ){
				try{
					f.get();
				}
				catch ( final InterruptedException | ExecutionException  e ){
					System.err.println(e.getMessage());
				}
			}
			
			afterRun();
			try {
				service.awaitTermination(5, TimeUnit.MINUTES);
			} catch (InterruptedException e) {
				System.err.println(e.getMessage());
			}
			return;
		}
		if (!outputs.isEmpty()){ // no inputs but maybe only outputs
			beforeRun();
						
			while (running) {
				if (Thread.currentThread().isInterrupted()) break;
				Map<String, Element> data = new HashMap<>();
				process(data);
				
				for (String key : data.keySet())
					newOutput(key,data.get(key));
			}
			
			afterRun();	
			return;
		}
	}

	protected void afterRun() {		
	}

	protected void beforeRun() {		
	}
	
	/**
	 * Method to be overwritten by children of this class.
	 * @param data - data to process
	 */
	public abstract void process(Map<String, Element> data);

}
