#pragma once
#include "stdafx.h"

class WorkerPool
{
public:
	uint32_t JobSize = 0; // Public so that you can set it to 0 early and cancel?
	// Second argument is worker id
	std::function<void(int,int)> WorkerFunction;

	WorkerPool(uint32_t NumWorkers) : wPool(NumWorkers)
	{
		IsAlive = true;
		WorkersCount = NumWorkers;

		WorkerLocks = new CV[WorkersCount];

		for (int id = 0; id < WorkersCount; id++)
		{
			wPool[id] = std::thread([id, this]() // worker code
			{
				WorkerLocks[id].Wait();

				while (IsAlive)
				{
					while (true)
					{
						int widx = JobIDX.fetch_add(1);
						if (widx >= JobSize)
						{
							// Only signal when the last worker finished
							if (ActiveWorkers.fetch_sub(1) == 1) MainLock.Signal();
							break;
						}

						WorkerFunction(widx,id);
					}

					WorkerLocks[id].Wait();
				}

				// Closing the thread. If this is the last worker to close, signal.
				if(ActiveWorkers.fetch_sub(1) == 1) MainLock.Signal();
			});
		}
	}
	
	~WorkerPool()
	{
		IsAlive = false;
		// Wake up all workers so that they terminate correctly
		ActiveWorkers = WorkersCount;
		for (int id = 0; id < WorkersCount; id++)
			WorkerLocks[id].Signal();

		MainLock.Wait();

		delete[] WorkerLocks;

		for (int id = 0; id < WorkersCount; id++)
			wPool[id].join();
	}

	void Dispatch()
	{
		JobIDX = 0;
		ActiveWorkers = WorkersCount;
		for (int id = 0; id < WorkersCount; id++)
			WorkerLocks[id].Signal();

		MainLock.Wait();
	}

	struct CV
	{
		std::mutex m;
		std::condition_variable cv;

		void Wait()
		{
			std::unique_lock<std::mutex> wlock(m);
			cv.wait(wlock);
		}

		void Signal()
		{
			cv.notify_one();
		}

		void SignalAll()
		{
			cv.notify_all();
		}
	};
private:
	std::vector<std::thread> wPool;

	int WorkersCount;
	std::atomic<int> JobIDX;
	std::atomic<int> ActiveWorkers;
	CV* WorkerLocks;
	CV MainLock;
	bool IsAlive;
};