package org.lemming.pipeline;

import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import org.lemming.interfaces.Element;

/**
 * multi threaded modules using Callables @see java.util.concurrent.Callables.
 * 
 * @author Ronny Sczech
 *
 */
public abstract class MultiRunModule extends AbstractModule{
	
	protected final ConcurrentLinkedQueue<Integer> counterList = new ConcurrentLinkedQueue<>();
	
	protected MultiRunModule(){
	}
	
	@Override
	public void run() {
		if (!inputs.isEmpty() && !outputs.isEmpty()) { // first check for existing inputs
			if (inputs.keySet().iterator().hasNext() && iterator==null)
				iterator = inputs.keySet().iterator().next();
			while (inputs.get(iterator).isEmpty())
				pause(10);
			beforeRun();
			
			final ArrayList<Future<Void>> futures = new ArrayList<>();

			for (int taskNum = 0; taskNum < numThreads; ++taskNum) {

				final Callable<Void> r = new Callable<Void>() {

					@Override
					public Void call() {
	                    while (running) {
	                        if (Thread.currentThread().isInterrupted())
	                            break;
	                        Element data = nextInput();
	                        if (data != null)
	                            newOutput(processData(data));
	                        else
	                            pause(10);
	                    }
	                    return null;
					}
				};
				futures.add(service.submit(r));
			}

			for (final Future<Void> f : futures) {
				try {
					f.get();
				} catch (final InterruptedException | ExecutionException e) {
					System.err.println(getClass().getSimpleName()+e.getMessage());
					e.printStackTrace();
				}
			}
			afterRun();
			return;
		}
		if (!inputs.isEmpty()) { // first check for existing inputs
			if (inputs.keySet().iterator().hasNext() && iterator==null)
				iterator = inputs.keySet().iterator().next();
			while (inputs.get(iterator).isEmpty())
				pause(10);
			beforeRun();
			
			final ArrayList<Future<Void>> futures = new ArrayList<>();

			for (int taskNum = 0; taskNum < numThreads; ++taskNum) {

				final Callable<Void> r = new Callable<Void>() {

					@Override
					public Void call() {
	                    while (running) {
	                        if (Thread.currentThread().isInterrupted())
	                            break;
	                        Element data = nextInput();
	                        if (data != null)
	                            processData(data);
	                        else pause(10);
	                    }
	                    return null;
					}
                };
				futures.add(service.submit(r));
			}

			for (final Future<Void> f : futures) {
				try {
					f.get();
				} catch (final InterruptedException | ExecutionException e) {
					e.printStackTrace();
				}
			}
			afterRun();
			return;
		}
		if (!outputs.isEmpty()) { // only output
			beforeRun();
			while (running) {
				Element data = processData(null);
				newOutput(data);
			}
			afterRun();
		}
	}

	protected void afterRun() {		
	}

	protected void beforeRun() {
		start = System.currentTimeMillis();
		running = true;
	}
}
