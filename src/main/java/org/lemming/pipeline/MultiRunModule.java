package org.lemming.pipeline;

import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import org.lemming.interfaces.Element;

public abstract class MultiRunModule extends AbstractModule{
	
	public MultiRunModule(){
	}
	
	@Override
	public void run() {

		if (!inputs.isEmpty() && !outputs.isEmpty()) { // first check for existing inputs
			if (inputs.keySet().iterator().hasNext() && iterator==null)
				iterator = inputs.keySet().iterator().next();
			beforeRun();
			if (inputs.get(iterator) != null) {
				while (inputs.get(iterator).isEmpty())
					pause(10);

				final ArrayList<Future<Void>> futures = new ArrayList<>();

				for (int taskNum = 0; taskNum < getNumThreads(); ++taskNum) {

					final Callable<Void> r = new Callable<Void>() {

						@Override
						public Void call() {
							while (running) {
								if (Thread.currentThread().isInterrupted())
									break;
								Element data = nextInput();
								if (data != null) 
									newOutput(process(data));
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
						System.err.println(e.getMessage());
					}
				}

				try {
					service.awaitTermination(5, TimeUnit.MINUTES);
				} catch (InterruptedException e) {
					System.err.println(e.getMessage());
				}
			}
			afterRun();
			return;
		}
		if (!inputs.isEmpty()) { // first check for existing inputs
			if (inputs.keySet().iterator().hasNext() && iterator==null)
				iterator = inputs.keySet().iterator().next();
			beforeRun();
			if (inputs.get(iterator) != null) {
				while (inputs.get(iterator).isEmpty())
					pause(10);

				final ArrayList<Future<Void>> futures = new ArrayList<>();

				for (int taskNum = 0; taskNum < getNumThreads(); ++taskNum) {

					final Callable<Void> r = new Callable<Void>() {

						@Override
						public Void call() {
							while (running) {
								if (Thread.currentThread().isInterrupted())
									break;
								Element data = nextInput();
								if (data != null) 
									process(data);
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
						System.err.println(e.getMessage());
					}
				}

				try {
					service.awaitTermination(5, TimeUnit.MINUTES);
				} catch (InterruptedException e) {
					System.err.println(e.getMessage());
				}
			}
			afterRun();
			return;
		}
		if (!outputs.isEmpty()) { // no inputs but maybe only output
			beforeRun();
			while (running) {
				if (Thread.currentThread().isInterrupted())
					break;
				Element data = process(null);
				newOutput(data);
			}
			afterRun();
			return;
		}
	}

	protected void afterRun() {		
	}

	protected void beforeRun() {		
	}
}
