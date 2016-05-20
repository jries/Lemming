package org.lemming.modules;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import org.lemming.interfaces.Element;
import org.lemming.pipeline.FrameElements;
import org.lemming.pipeline.LocalizationPrecision3D;

import net.imglib2.type.numeric.RealType;

public abstract class CPU_Fitter<T extends RealType<T>> extends Fitter<T> {
	
	private final ConcurrentLinkedQueue<Integer> counterList = new ConcurrentLinkedQueue<>();

	protected CPU_Fitter(int halfkernel) {
		super(halfkernel);
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
				if (Thread.currentThread().isInterrupted())
					break;
				Element data = processData(null);
				newOutput(data);
			}
			afterRun();
		}
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public Element processData(Element data) {
		FrameElements<T> fe = (FrameElements<T>) data;

		if (fe.isLast()) {
			cancel();
			process(fe);
			return null;
		}
		process(fe);
		return null;
	}

	private void process(FrameElements<T> data) {
		List<Element> res = fit(data.getList(), data.getFrame(), size);
		counterList.add(res.size());
		for (Element l:res) newOutput(l);
	}

	private void afterRun() {
		Integer cc = 0;
		for (Integer i : counterList)
			cc += i;
		LocalizationPrecision3D lastLoc = new LocalizationPrecision3D(-1, -1, -1, 0, 0, 0, 1, 1L);
		lastLoc.setLast(true);
		newOutput(lastLoc);
		System.out.println("Fitting of " + cc + " elements done in " + (System.currentTimeMillis() - start) + "ms");
	}

}
