package org.lemming.tools;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

public class WaitForChangeListener implements ChangeListener {

	private long delay;
	private Runnable command;
	private ScheduledExecutorService executor = Executors.newSingleThreadScheduledExecutor();
	private ScheduledFuture< ? > future;
	

	public WaitForChangeListener(long delay, Runnable command) {
		this.delay = delay;
		this.command = command;
	}

	@Override
	public void stateChanged(ChangeEvent arg0) {
		if (future != null && !future.isDone()) {
			future.cancel(false);
		}
		future = executor.schedule(command, delay, TimeUnit.MILLISECONDS);
	}

}
